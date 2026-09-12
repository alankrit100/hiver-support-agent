"""Ensemble model selection: bge_lr + Sarvam few-shot, weighted vote.

Selects the blend weight `weight_bge` (bge_lr proba vs. Sarvam pseudo-proba)
via 5-fold stratified CV on dev-170 ONLY -- same splits/seed as
scripts/run_v4_cv.py, so results are directly comparable.

Leakage rules (unchanged from the base V4 protocol):
- Never touches data/test_set.jsonl (frozen 45).
- bge_lr is refit per-fold on the training fold only (no leakage).
- Sarvam predictions are precomputed and fixed (few-shot, not trained on
  dev labels), reused across folds -- identical assumption to
  scripts/run_v4_cv.py's sarvam_fold_metrics.
- weight_bge is chosen by CV mean macro-F1 on dev-170 only.

Prerequisite:
    python scripts/run_v4_cv.py --with-sarvam
    (caches reports/v4_sarvam_dev170.json and data/v4_bge_m3_dev170.npy)

Usage (run from repo root):
    python scripts/run_v4_ensemble_cv.py

Outputs:
    reports/v4_ensemble_cv_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from v4_classifier import LABELS, build_bge_lr  # noqa: E402
from ensemble_classifier import ensemble_predict  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "train_set.jsonl"
EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.npy"
EMB_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.ids.json"
SARVAM_PREDS_PATH = REPO_ROOT / "reports" / "v4_sarvam_dev170.json"
REPORT_PATH = REPO_ROOT / "reports" / "v4_ensemble_cv_results.json"

N_SPLITS = 5
SEED = 42
BGE_LR_C = 2.0  # winning hyperparameter from reports/v4_cv_results.json
WEIGHT_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def load_dev() -> list:
    rows = []
    with open(DEV_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == 170, f"expected 170 dev examples, got {len(rows)}"
    return rows


def fold_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                  zero_division=0)),
    }


def summarize(folds: list) -> dict:
    acc = np.array([f["accuracy"] for f in folds])
    mac = np.array([f["macro_f1"] for f in folds])
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(mac)),
        "macro_f1_std": float(np.std(mac)),
        "folds": folds,
    }


def main() -> None:
    if not SARVAM_PREDS_PATH.exists():
        sys.exit(
            f"Missing {SARVAM_PREDS_PATH}. Run first:\n"
            f"    python scripts/run_v4_cv.py --with-sarvam"
        )
    if not EMB_PATH.exists() or not EMB_IDS_PATH.exists():
        sys.exit(
            "Missing cached BGE-M3 dev embeddings. Run first:\n"
            "    python scripts/run_v4_cv.py"
        )

    dev = load_dev()
    y = [r["labeled_intent"] for r in dev]
    ids = [r["tweet_id"] for r in dev]

    cached_ids = json.loads(EMB_IDS_PATH.read_text())
    assert cached_ids == ids, "dev embedding cache order mismatch vs train_set.jsonl"
    X_emb = np.load(EMB_PATH)

    sarvam_preds = json.loads(SARVAM_PREDS_PATH.read_text())
    sarvam_by_id = {p["tweet_id"]: p for p in sarvam_preds}
    sarvam_pred_list = [sarvam_by_id[tid]["pred_intent"] for tid in ids]
    sarvam_conf_list = [sarvam_by_id[tid]["confidence"] for tid in ids]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_emb, y))

    results = {}
    for weight_bge in WEIGHT_GRID:
        folds = []
        for fold_i, (tr, va) in enumerate(splits):
            clf = build_bge_lr(C=BGE_LR_C)
            clf.fit(X_emb[tr], [y[i] for i in tr])
            bge_proba = clf.predict_proba(X_emb[va])
            # align predict_proba columns to canonical LABELS order
            col_for_label = {c: i for i, c in enumerate(clf.classes_)}
            bge_proba_aligned = np.zeros((len(va), len(LABELS)))
            for j, label in enumerate(LABELS):
                if label in col_for_label:
                    bge_proba_aligned[:, j] = bge_proba[:, col_for_label[label]]

            preds = []
            for row_i, idx in enumerate(va):
                out = ensemble_predict(
                    bge_proba_aligned[row_i],
                    sarvam_pred_list[idx],
                    sarvam_conf_list[idx],
                    weight_bge,
                )
                preds.append(out["intent"])

            m = fold_metrics([y[i] for i in va], preds)
            m["fold"] = fold_i
            folds.append(m)
            print(f"[ENSEMBLE-CV] weight_bge={weight_bge} fold {fold_i}: "
                  f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
        results[str(weight_bge)] = summarize(folds)

    best_weight, best_summary = max(results.items(), key=lambda kv: kv[1]["macro_f1_mean"])

    report = {
        "protocol": {
            "dev_size": 170,
            "n_splits": N_SPLITS,
            "seed": SEED,
            "selection_metric": "macro_f1_mean",
            "bge_lr_C": BGE_LR_C,
            "frozen_test_touched": False,
            "note": "weight_bge=1.0 reduces to bge_lr alone; weight_bge=0.0 reduces to Sarvam alone",
        },
        "by_weight": results,
        "selection": {
            "weight_bge": float(best_weight),
            "macro_f1_mean": best_summary["macro_f1_mean"],
        },
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print("ENSEMBLE CV SUMMARY (dev-170, 5-fold stratified)")
    print("=" * 70)
    for weight_bge in WEIGHT_GRID:
        s = results[str(weight_bge)]
        print(f"weight_bge={weight_bge:.1f}  acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}  "
              f"macroF1={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}")
    print(f"\nBEST: weight_bge={best_weight} (macroF1={best_summary['macro_f1_mean']:.3f})")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
