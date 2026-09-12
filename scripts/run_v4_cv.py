"""V4 model selection: 5-fold stratified CV on the 170-example dev set.

Leakage rules (brief Section 11):
- Trains and selects ONLY on data/train_set.jsonl (170 dev examples).
- NEVER touches data/test_set.jsonl (frozen 45).
- TF-IDF vectorizer lives inside the sklearn Pipeline so vocabulary
  statistics never leak across fold boundaries.
- BGE-M3 embeddings are fixed precomputed features (no label use).

Selection metric: mean macro-F1 (classes are imbalanced).

Usage (run from repo root):
    python scripts/run_v4_cv.py                    # supervised candidates
    python scripts/run_v4_cv.py --with-sarvam      # + Sarvam baseline (170 LLM calls)
    python scripts/run_v4_cv.py --rebuild-embeddings  # force re-encode BGE-M3

Outputs:
    data/v4_bge_m3_dev170.npy      # cached dev embeddings (tweet_id order in .ids.json)
    reports/v4_cv_results.json     # full per-fold metrics for every candidate
    reports/v4_sarvam_dev170.json  # Sarvam preds on dev-170 (only with --with-sarvam)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from v4_classifier import CANDIDATES, LABELS  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "train_set.jsonl"
EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.npy"
EMB_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.ids.json"
REPORT_PATH = REPO_ROOT / "reports" / "v4_cv_results.json"
SARVAM_PREDS_PATH = REPO_ROOT / "reports" / "v4_sarvam_dev170.json"

N_SPLITS = 5
SEED = 42
BGE_MODEL = "BAAI/bge-m3"


def load_dev(path: Path = DEV_PATH) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == 170, f"expected 170 dev examples, got {len(rows)}"
    return rows


def get_dev_embeddings(texts: list, ids: list, rebuild: bool = False) -> np.ndarray:
    """Encode dev texts with BGE-M3 once; cache keyed on tweet_id order."""
    if EMB_PATH.exists() and EMB_IDS_PATH.exists() and not rebuild:
        cached_ids = json.loads(EMB_IDS_PATH.read_text())
        if cached_ids == ids:
            print(f"[CV] Loading cached BGE-M3 embeddings from {EMB_PATH}")
            return np.load(EMB_PATH)
        print("[CV] Cache ID mismatch, re-encoding...")
    from sentence_transformers import SentenceTransformer

    print(f"[CV] Encoding {len(texts)} dev texts with {BGE_MODEL}...")
    model = SentenceTransformer(BGE_MODEL)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    np.save(EMB_PATH, emb)
    EMB_IDS_PATH.write_text(json.dumps(ids))
    print(f"[CV] Saved embeddings {emb.shape} to {EMB_PATH}")
    return emb


def fold_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                  zero_division=0)),
        "per_intent_f1": {
            label: float(v)
            for label, v in zip(
                LABELS,
                f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
            )
        },
    }


def summarize(folds: list) -> dict:
    acc = np.array([f["accuracy"] for f in folds])
    mac = np.array([f["macro_f1"] for f in folds])
    per = {
        label: float(np.mean([f["per_intent_f1"][label] for f in folds]))
        for label in LABELS
    }
    per_std = {
        label: float(np.std([f["per_intent_f1"][label] for f in folds]))
        for label in LABELS
    }
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(mac)),
        "macro_f1_std": float(np.std(mac)),
        "per_intent_f1_mean": per,
        "per_intent_f1_std": per_std,
        "folds": folds,
    }


def run_supervised_cv(texts, y, X_emb) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(texts, y))
    results = {}
    for name, spec in CANDIDATES.items():
        results[name] = {}
        for C in spec["grid"]:
            folds = []
            for fold_i, (tr, va) in enumerate(splits):
                clf = spec["builder"](C=C)
                if spec["features"] == "text":
                    clf.fit([texts[i] for i in tr], [y[i] for i in tr])
                    pred = clf.predict([texts[i] for i in va])
                else:
                    clf.fit(X_emb[tr], [y[i] for i in tr])
                    pred = clf.predict(X_emb[va])
                m = fold_metrics([y[i] for i in va], list(pred))
                m["fold"] = fold_i
                folds.append(m)
                print(f"[CV] {name} C={C} fold {fold_i}: "
                      f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
            results[name][str(C)] = summarize(folds)
    return results


def run_sarvam_dev(texts, ids) -> dict:
    """Run Sarvam few-shot baseline ONCE over dev-170; reuse preds per fold."""
    if SARVAM_PREDS_PATH.exists():
        print(f"[CV] Loading cached Sarvam dev preds from {SARVAM_PREDS_PATH}")
        return json.loads(SARVAM_PREDS_PATH.read_text())
    from intent_classifier import IntentClassifier
    from llm_client import LLMClient

    # Pinned to Sarvam explicitly: this is a frozen, already-reported baseline
    # number. Since GROQ_API_KEY may also be set (added later as a more
    # reliable backend for reply generation/judging), LLMClient()'s
    # auto-detect would silently switch to Groq here and produce a different,
    # unreproducible number under the same "Sarvam baseline" label.
    clf = IntentClassifier(llm_client=LLMClient(provider="sarvam"))
    preds = []
    t0 = time.time()
    for i, (tid, text) in enumerate(zip(ids, texts)):
        r = clf.classify(text)
        preds.append({"tweet_id": tid, "pred_intent": r["intent"],
                      "confidence": r["confidence"]})
        if (i + 1) % 10 == 0:
            print(f"[CV] Sarvam {i + 1}/170 ({time.time() - t0:.0f}s elapsed)")
    SARVAM_PREDS_PATH.parent.mkdir(exist_ok=True)
    SARVAM_PREDS_PATH.write_text(json.dumps(preds, indent=2))
    return preds


def sarvam_fold_metrics(y, sarvam_preds) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = []
    for fold_i, (_, va) in enumerate(skf.split(y, y)):
        yt = [y[i] for i in va]
        yp = [sarvam_preds[i] for i in va]
        m = fold_metrics(yt, yp)
        m["fold"] = fold_i
        folds.append(m)
    return {"dev_predictions": summarize(folds)}


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 5-fold CV on dev-170")
    parser.add_argument("--with-sarvam", action="store_true",
                        help="Also run Sarvam baseline once over dev-170")
    parser.add_argument("--rebuild-embeddings", action="store_true",
                        help="Force re-encode BGE-M3 embeddings")
    args = parser.parse_args()

    dev = load_dev()
    texts = [r["text"] for r in dev]
    y = [r["labeled_intent"] for r in dev]
    ids = [r["tweet_id"] for r in dev]

    X_emb = get_dev_embeddings(texts, ids, rebuild=args.rebuild_embeddings)

    results = {
        "protocol": {
            "dev_size": len(dev),
            "n_splits": N_SPLITS,
            "seed": SEED,
            "selection_metric": "macro_f1_mean",
            "input": "text field only",
            "frozen_test_touched": False,
        },
        "supervised": run_supervised_cv(texts, y, X_emb),
    }

    if args.with_sarvam:
        sarvam = run_sarvam_dev(texts, ids)
        pred_by_id = {p["tweet_id"]: p["pred_intent"] for p in sarvam}
        yp = [pred_by_id[tid] for tid in ids]
        results["sarvam_baseline"] = sarvam_fold_metrics(y, yp)

    # Winner = best mean macro-F1 among supervised (Sarvam is a reference).
    best = (None, None, -1.0)
    for name, grids in results["supervised"].items():
        for c_str, summary in grids.items():
            if summary["macro_f1_mean"] > best[2]:
                best = (name, c_str, summary["macro_f1_mean"])
    results["selection"] = {"winner": best[0], "C": best[1],
                            "macro_f1_mean": best[2]}

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    print("V4 CV SUMMARY (dev-170, 5-fold stratified, metric = macro-F1)")
    print("=" * 70)
    for name, grids in results["supervised"].items():
        for c_str, s in grids.items():
            print(f"{name:10s} C={c_str:>4s}  "
                  f"acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}  "
                  f"macroF1={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}")
    if "sarvam_baseline" in results:
        s = results["sarvam_baseline"]["dev_predictions"]
        print(f"{'sarvam':10s} {'few-shot':>7s}  "
              f"acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}  "
              f"macroF1={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}")
    print(f"\nWINNER: {best[0]} C={best[1]} (macroF1={best[2]:.3f})")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
