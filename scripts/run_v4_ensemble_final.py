"""Ensemble final evaluation: fit bge_lr on dev-170, blend with Sarvam,
evaluate ONCE on frozen test-45.

Protocol (mirrors scripts/run_v4_final.py):
- The blend weight comes from reports/v4_ensemble_cv_results.json
  "selection" (CV on dev-170 only). This script never selects or tunes.
- bge_lr is fit on data/train_set.jsonl (170) only.
- data/test_set.jsonl (45) is used for prediction ONLY.
- Sarvam predictions on the frozen test-45 are needed for the blend. They
  are fetched live ONCE (45 API calls) and cached to
  reports/v4_sarvam_test45.json so re-running this script doesn't re-call
  the API or accidentally get different predictions on a re-run.
- Run EXACTLY ONCE per Sarvam-call sense. Do not re-run after editing
  anything based on the score.

Usage (run from repo root):
    python scripts/run_v4_ensemble_final.py

Outputs:
    reports/v4_sarvam_test45.json           # cached Sarvam preds on test-45
    reports/v4_ensemble_final_results.json  # accuracy, macro F1, per-intent,
                                            # confusion matrix, per-example preds
    reports/v4_ensemble_confusion_matrix.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from v4_classifier import LABELS, build_bge_lr  # noqa: E402
from ensemble_classifier import ensemble_predict  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "train_set.jsonl"
TEST_PATH = REPO_ROOT / "data" / "test_set.jsonl"
DEV_EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.npy"
DEV_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.ids.json"
TEST_EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_test45.npy"
TEST_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_test45.ids.json"
ENSEMBLE_CV_PATH = REPO_ROOT / "reports" / "v4_ensemble_cv_results.json"
BGE_FINAL_PATH = REPO_ROOT / "reports" / "v4_final_results.json"
SARVAM_TEST_PATH = REPO_ROOT / "reports" / "v4_sarvam_test45.json"
FINAL_REPORT_PATH = REPO_ROOT / "reports" / "v4_ensemble_final_results.json"
CM_PNG_PATH = REPO_ROOT / "reports" / "v4_ensemble_confusion_matrix.png"

BGE_LR_C = 2.0  # winning hyperparameter from reports/v4_cv_results.json


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_sarvam_test_preds(test_texts: list, test_ids: list) -> dict:
    """Fetch (and cache) Sarvam few-shot predictions on the frozen test-45."""
    if SARVAM_TEST_PATH.exists():
        print(f"[ENSEMBLE-FINAL] Loading cached Sarvam test preds from {SARVAM_TEST_PATH}")
        cached = json.loads(SARVAM_TEST_PATH.read_text())
        if [p["tweet_id"] for p in cached] == test_ids:
            return {p["tweet_id"]: p for p in cached}
        print("[ENSEMBLE-FINAL] Cache ID mismatch, re-fetching...")

    from intent_classifier import IntentClassifier
    from llm_client import LLMClient

    # Pinned to Sarvam explicitly -- see the matching comment in
    # scripts/run_v4_cv.py. This is the frozen 68.9%/0.691 ensemble result;
    # GROQ_API_KEY being set for reply generation must not silently change
    # which model produces this number on a re-run.
    clf = IntentClassifier(llm_client=LLMClient(provider="sarvam"))
    preds = []
    t0 = time.time()
    for i, (tid, text) in enumerate(zip(test_ids, test_texts)):
        r = clf.classify(text)
        preds.append({"tweet_id": tid, "pred_intent": r["intent"],
                      "confidence": r["confidence"]})
        if (i + 1) % 10 == 0:
            print(f"[ENSEMBLE-FINAL] Sarvam {i + 1}/{len(test_ids)} "
                  f"({time.time() - t0:.0f}s elapsed)")
    SARVAM_TEST_PATH.parent.mkdir(exist_ok=True)
    SARVAM_TEST_PATH.write_text(json.dumps(preds, indent=2))
    return {p["tweet_id"]: p for p in preds}


def main() -> None:
    selection = json.loads(ENSEMBLE_CV_PATH.read_text())["selection"]
    weight_bge = selection["weight_bge"]
    print(f"[ENSEMBLE-FINAL] Blend weight from CV: weight_bge={weight_bge} "
          f"(CV macroF1={selection['macro_f1_mean']:.3f})")

    # --- Load dev-170 + cached embeddings ---
    dev = load_jsonl(DEV_PATH)
    assert len(dev) == 170, f"expected 170 dev, got {len(dev)}"
    dev_ids = [r["tweet_id"] for r in dev]
    assert json.loads(DEV_IDS_PATH.read_text()) == dev_ids, \
        "dev embedding cache does not match train_set.jsonl order"
    X_dev = np.load(DEV_EMB_PATH)
    y_dev = [r["labeled_intent"] for r in dev]

    # --- Load frozen test-45 (predict ONLY) ---
    test = load_jsonl(TEST_PATH)
    assert len(test) == 45, f"expected 45 test, got {len(test)}"
    test_ids = [r["tweet_id"] for r in test]
    assert set(test_ids).isdisjoint(dev_ids), "LEAKAGE: test/dev overlap!"
    y_true = [r["labeled_intent"] for r in test]
    test_texts = [r["text"] for r in test]

    assert json.loads(TEST_IDS_PATH.read_text()) == test_ids, \
        "test embedding cache does not match test_set.jsonl order"
    X_test = np.load(TEST_EMB_PATH)

    # --- Fit bge_lr on ALL 170 dev, predict_proba on test ---
    clf = build_bge_lr(C=BGE_LR_C)
    clf.fit(X_dev, y_dev)
    bge_proba = clf.predict_proba(X_test)
    col_for_label = {c: i for i, c in enumerate(clf.classes_)}
    bge_proba_aligned = np.zeros((len(test_ids), len(LABELS)))
    for j, label in enumerate(LABELS):
        if label in col_for_label:
            bge_proba_aligned[:, j] = bge_proba[:, col_for_label[label]]

    # --- Sarvam predictions on test-45 (live once, then cached) ---
    sarvam_by_id = get_sarvam_test_preds(test_texts, test_ids)

    # --- Blend ---
    y_pred, confidence = [], []
    for row_i, tid in enumerate(test_ids):
        sp = sarvam_by_id[tid]
        out = ensemble_predict(bge_proba_aligned[row_i], sp["pred_intent"],
                               sp["confidence"], weight_bge)
        y_pred.append(out["intent"])
        confidence.append(out["confidence"])

    # --- Metrics ---
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                             zero_division=0))
    p = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    support = {label: int(y_true.count(label)) for label in LABELS}
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    examples = [
        {"tweet_id": tid, "true_intent": t, "pred_intent": pr, "confidence": c,
         "correct": t == pr}
        for tid, t, pr, c in zip(test_ids, y_true, y_pred, confidence)
    ]
    correct = sum(e["correct"] for e in examples)

    bge_only = json.loads(BGE_FINAL_PATH.read_text()) if BGE_FINAL_PATH.exists() else None

    report = {
        "protocol": {
            "model": "ensemble(bge_lr, sarvam_few_shot)",
            "weight_bge": weight_bge,
            "bge_lr_C": BGE_LR_C,
            "fit_on": "dev-170 (all)",
            "evaluated_on": "frozen test-45 (once)",
            "cv_macro_f1_mean": selection["macro_f1_mean"],
        },
        "accuracy": acc,
        "macro_f1": macro_f1,
        "correct": correct,
        "total": len(y_true),
        "per_intent": {
            label: {"precision": float(p[i]), "recall": float(r[i]),
                    "f1": float(f[i]), "support": support[label]}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
        "examples": examples,
        "comparison_vs_bge_lr_alone": {
            "bge_lr_accuracy": bge_only["accuracy"] if bge_only else None,
            "bge_lr_macro_f1": bge_only["macro_f1"] if bge_only else None,
            "ensemble_accuracy": acc,
            "ensemble_macro_f1": macro_f1,
            "accuracy_delta": (acc - bge_only["accuracy"]) if bge_only else None,
            "macro_f1_delta": (macro_f1 - bge_only["macro_f1"]) if bge_only else None,
        },
    }
    FINAL_REPORT_PATH.parent.mkdir(exist_ok=True)
    FINAL_REPORT_PATH.write_text(json.dumps(report, indent=2))

    # --- Confusion matrix plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"V4 ensemble (weight_bge={weight_bge}): acc={acc:.3f} "
                 f"macroF1={macro_f1:.3f} ({correct}/45)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PNG_PATH, dpi=150)

    print("\n" + "=" * 70)
    print(f"ENSEMBLE FINAL (fit dev-170 -> frozen test-45, run once)")
    print(f"accuracy={acc:.3f} ({correct}/45)  macroF1={macro_f1:.3f}")
    if bge_only:
        print(f"vs bge_lr alone: accuracy={bge_only['accuracy']:.3f} "
              f"macroF1={bge_only['macro_f1']:.3f}")
    for label in LABELS:
        d = report["per_intent"][label]
        print(f"  {label:28s} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"F1={d['f1']:.3f} n={d['support']}")
    print(f"Report: {FINAL_REPORT_PATH}\nPlot:   {CM_PNG_PATH}")


if __name__ == "__main__":
    main()
