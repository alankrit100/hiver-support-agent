"""V4 final evaluation: fit winner on dev-170, evaluate ONCE on frozen test-45.

Protocol (brief Sections 3, 11):
- Winner + hyperparameters come from reports/v4_cv_results.json "selection"
  (CV on dev-170 only). This script never selects, tunes, or compares.
- Fits on data/train_set.jsonl (170) only.
- data/test_set.jsonl (45) is loaded for prediction ONLY: asserted to be
  45 rows with zero tweet_id overlap vs dev. Never fit, never tuned on.
- Run EXACTLY ONCE. Do not re-run after editing anything based on the score.

Usage (run from repo root):
    python scripts/run_v4_final.py

Outputs:
    data/v4_bge_m3_test45.npy          # cached test embeddings (ids in .ids.json)
    reports/v4_final_results.json      # accuracy, macro F1, per-intent P/R/F1,
                                       # confusion matrix, per-example predictions
    reports/v4_confusion_matrix.png    # confusion matrix heatmap
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from v4_classifier import CANDIDATES, LABELS  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "train_set.jsonl"
TEST_PATH = REPO_ROOT / "data" / "test_set.jsonl"
DEV_EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.npy"
DEV_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.ids.json"
TEST_EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_test45.npy"
TEST_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_test45.ids.json"
CV_REPORT_PATH = REPO_ROOT / "reports" / "v4_cv_results.json"
FINAL_REPORT_PATH = REPO_ROOT / "reports" / "v4_final_results.json"
CM_PNG_PATH = REPO_ROOT / "reports" / "v4_confusion_matrix.png"
BGE_MODEL = "BAAI/bge-m3"


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_texts(texts: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[FINAL] Encoding {len(texts)} texts with {BGE_MODEL}...")
    model = SentenceTransformer(BGE_MODEL)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def main() -> None:
    # --- Winner comes from CV selection, not from this script ---
    selection = json.loads(CV_REPORT_PATH.read_text())["selection"]
    name, C = selection["winner"], float(selection["C"])
    spec = CANDIDATES[name]
    print(f"[FINAL] Winner from CV: {name} C={C} "
          f"(CV macroF1={selection['macro_f1_mean']:.3f})")
    assert spec["features"] == "embedding", \
        f"expected embedding winner, got {spec['features']}"

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

    # --- Embed test (separate cache, never pooled with dev) ---
    if TEST_EMB_PATH.exists() and TEST_IDS_PATH.exists():
        assert json.loads(TEST_IDS_PATH.read_text()) == test_ids, \
            "test embedding cache does not match test_set.jsonl order"
        print(f"[FINAL] Loading cached test embeddings from {TEST_EMB_PATH}")
        X_test = np.load(TEST_EMB_PATH)
    else:
        X_test = encode_texts(test_texts)
        np.save(TEST_EMB_PATH, X_test)
        TEST_IDS_PATH.write_text(json.dumps(test_ids))

    # --- Fit on ALL 170 dev, predict test ONCE ---
    clf = spec["builder"](C=C)
    clf.fit(X_dev, y_dev)
    y_pred = list(clf.predict(X_test))
    proba = clf.predict_proba(X_test)
    confidence = [float(np.max(row)) for row in proba]

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

    report = {
        "protocol": {
            "model": name, "C": C,
            "fit_on": "dev-170 (all)",
            "evaluated_on": "frozen test-45 (once)",
            "input": "text field only",
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
    ax.set_title(f"V4 final ({name} C={C}): acc={acc:.3f} macroF1={macro_f1:.3f} "
                 f"({correct}/45)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PNG_PATH, dpi=150)

    print("\n" + "=" * 70)
    print(f"V4 FINAL (fit dev-170 -> frozen test-45, run once)")
    print(f"accuracy={acc:.3f} ({correct}/45)  macroF1={macro_f1:.3f}")
    for label in LABELS:
        d = report["per_intent"][label]
        print(f"  {label:28s} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"F1={d['f1']:.3f} n={d['support']}")
    print(f"Report: {FINAL_REPORT_PATH}\nPlot:   {CM_PNG_PATH}")


if __name__ == "__main__":
    main()
