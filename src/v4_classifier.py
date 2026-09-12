"""V4 supervised intent classifiers.

Candidates required by the V4 brief (Section 3):
  1. TF-IDF + Logistic Regression
  2. BGE-M3 embeddings + Logistic Regression
  3. BGE-M3 embeddings + Linear SVM

Design notes (see DECISION_LOG.md after Step 11):
- Input is the ``text`` field ONLY (no thread_context). Decided with the
  user: cleanest option, no thread leakage risk, matches V1-V3 eval setup.
- All supervised candidates use ``class_weight="balanced"`` because the dev
  set is imbalanced (device_crash_freeze has only 13 dev examples).
- Canonical label order follows ``intent_definitions.json`` so confusion
  matrices are stable across runs.
- BGE-M3 embeddings are precomputed once by ``scripts/run_v4_cv.py`` and
  passed in as arrays; this module never touches the frozen test set.

Usage:
    from v4_classifier import LABELS, build_tfidf_lr, build_bge_lr, build_bge_svm
    clf = build_tfidf_lr(C=2.0)
    clf.fit(["my battery drains fast", ...], ["battery_drain_after_update", ...])
    clf.predict(["phone keeps freezing"])
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Canonical label order (matches src/intent_definitions.json).
LABELS = [
    "ios_update_issues",
    "battery_drain_after_update",
    "device_crash_freeze",
    "app_malfunction",
    "account_access",
    "unclear",
]

# Hyperparameter grids searched with 5-fold stratified CV on dev-170 only.
TFIDF_C_GRID = [0.5, 1.0, 2.0, 4.0]
BGE_LR_C_GRID = [0.5, 1.0, 2.0, 4.0]
BGE_SVM_C_GRID = [0.5, 1.0, 2.0]


def build_tfidf_lr(C: float = 2.0) -> Pipeline:
    """TF-IDF (word 1-2 grams) + Logistic Regression pipeline.

    Operates on raw text; the vectorizer is part of the pipeline so CV
    folds never leak vocabulary statistics across fold boundaries.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


def build_bge_lr(C: float = 1.0) -> LogisticRegression:
    """Logistic Regression on precomputed BGE-M3 embeddings."""
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
    )


def build_bge_svm(C: float = 1.0) -> CalibratedClassifierCV:
    """Linear SVM on precomputed BGE-M3 embeddings, calibrated for confidence.

    LinearSVC has no predict_proba, so wrap in CalibratedClassifierCV
    (sigmoid, inner cv=3) to produce intent_confidence for the V4 output
    schema and escalation policy.
    """
    base = LinearSVC(C=C, class_weight="balanced", max_iter=5000)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


# Registry used by scripts/run_v4_cv.py and scripts/run_v4_final.py.
# "features" indicates expected input: raw "text" or precomputed "embedding".
CANDIDATES = {
    "tfidf_lr": {"features": "text", "grid": TFIDF_C_GRID, "builder": build_tfidf_lr},
    "bge_lr": {"features": "embedding", "grid": BGE_LR_C_GRID, "builder": build_bge_lr},
    "bge_svm": {"features": "embedding", "grid": BGE_SVM_C_GRID, "builder": build_bge_svm},
}
