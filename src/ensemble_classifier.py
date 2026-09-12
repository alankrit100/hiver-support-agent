"""Ensemble of the V4 winner (BGE-M3 + Logistic Regression) and the Sarvam
few-shot LLM baseline.

Rationale (see reports/v4_cv_results.json): bge_lr and the Sarvam classifier
are trained/prompted very differently and their confusion matrices don't
fully overlap, so a weighted vote can pick up a few points without any new
data or model. This module only combines predictions that already exist
elsewhere in the repo -- it does not introduce a new modeling approach.

Sarvam has no per-class probabilities, only a single predicted intent +
a self-reported confidence. To combine it with bge_lr's predict_proba, we
turn the Sarvam prediction into a pseudo-distribution: `confidence` mass on
the predicted label, the remainder split uniformly over the other labels.
This is a heuristic, not a calibrated probability -- documented here so it
isn't mistaken for one elsewhere.
"""

import json
from pathlib import Path

import numpy as np

from v4_classifier import LABELS, build_bge_lr

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "train_set.jsonl"
DEV_EMB_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.npy"
DEV_IDS_PATH = REPO_ROOT / "data" / "v4_bge_m3_dev170.ids.json"
ENSEMBLE_CV_PATH = REPO_ROOT / "reports" / "v4_ensemble_cv_results.json"
BGE_LR_C = 2.0
BGE_MODEL_NAME = "BAAI/bge-m3"


def sarvam_to_distribution(pred_intent: str, confidence: float,
                            labels: list = LABELS) -> np.ndarray:
    """Turn a single Sarvam (intent, confidence) prediction into a pseudo-distribution."""
    confidence = max(0.0, min(1.0, float(confidence)))
    n = len(labels)
    dist = np.full(n, (1.0 - confidence) / max(n - 1, 1), dtype=np.float64)
    if pred_intent in labels:
        dist[labels.index(pred_intent)] = confidence
    else:
        # Unknown/invalid label from Sarvam: fall back to uniform.
        dist = np.full(n, 1.0 / n, dtype=np.float64)
    return dist


def combine_proba(bge_proba: np.ndarray, sarvam_dist: np.ndarray,
                   weight_bge: float) -> np.ndarray:
    """Weighted sum of two label distributions. weight_bge in [0, 1]."""
    return weight_bge * bge_proba + (1.0 - weight_bge) * sarvam_dist


def ensemble_predict(bge_proba_row: np.ndarray, sarvam_pred: str,
                      sarvam_confidence: float, weight_bge: float,
                      labels: list = LABELS) -> dict:
    """Combine one bge_lr probability row with one Sarvam prediction.

    Returns dict with keys: intent, confidence, combined_proba (list).
    """
    sarvam_dist = sarvam_to_distribution(sarvam_pred, sarvam_confidence, labels)
    combined = combine_proba(np.asarray(bge_proba_row, dtype=np.float64),
                              sarvam_dist, weight_bge)
    best_idx = int(np.argmax(combined))
    return {
        "intent": labels[best_idx],
        "confidence": float(combined[best_idx]),
        "combined_proba": combined.tolist(),
    }


class LiveEnsembleClassifier:
    """Production wrapper: this is the classifier the pipeline should use.

    Fits bge_lr once (on all 170 dev examples, reusing the cached BGE-M3
    embeddings -- cheap, no re-encoding) and blends it with a live Sarvam
    few-shot call per message, using the weight_bge selected by
    scripts/run_v4_ensemble_cv.py on dev-170 only.

    Encoding a new message requires loading the BAAI/bge-m3
    sentence-transformers model (same one used by scripts/run_v4_cv.py),
    which is a few hundred MB and lazy-loaded on first classify() call.
    """

    def __init__(self, llm_client=None, weight_bge: float = None):
        from intent_classifier import IntentClassifier  # local import: avoid

        # circular import at module load time (pipeline.py imports both).
        self._sarvam = IntentClassifier(llm_client=llm_client)
        self._bge_encoder = None
        self.weight_bge = weight_bge if weight_bge is not None else self._load_selected_weight()
        self._lr = self._fit_bge_lr()

    @staticmethod
    def _load_selected_weight() -> float:
        if not ENSEMBLE_CV_PATH.exists():
            raise RuntimeError(
                f"Missing {ENSEMBLE_CV_PATH}. Run first:\n"
                f"    python scripts/run_v4_ensemble_cv.py"
            )
        return json.loads(ENSEMBLE_CV_PATH.read_text())["selection"]["weight_bge"]

    @staticmethod
    def _fit_bge_lr():
        if not DEV_EMB_PATH.exists() or not DEV_IDS_PATH.exists():
            raise RuntimeError(
                "Missing cached BGE-M3 dev embeddings. Run first:\n"
                "    python scripts/run_v4_cv.py"
            )
        dev = []
        with open(TRAIN_PATH) as f:
            for line in f:
                if line.strip():
                    dev.append(json.loads(line))
        dev_ids = [r["tweet_id"] for r in dev]
        assert json.loads(DEV_IDS_PATH.read_text()) == dev_ids, \
            "dev embedding cache order mismatch vs train_set.jsonl"
        X_dev = np.load(DEV_EMB_PATH)
        y_dev = [r["labeled_intent"] for r in dev]
        clf = build_bge_lr(C=BGE_LR_C)
        clf.fit(X_dev, y_dev)
        return clf

    def _load_bge_encoder(self):
        if self._bge_encoder is None:
            from sentence_transformers import SentenceTransformer

            self._bge_encoder = SentenceTransformer(BGE_MODEL_NAME)
        return self._bge_encoder

    def classify(self, message: str) -> dict:
        """Classify one message. Returns dict with keys: intent, confidence, method."""
        encoder = self._load_bge_encoder()
        emb = encoder.encode([message], normalize_embeddings=True)
        proba_row = self._lr.predict_proba(np.asarray(emb, dtype=np.float32))[0]
        col_for_label = {c: i for i, c in enumerate(self._lr.classes_)}
        proba_aligned = np.zeros(len(LABELS))
        for j, label in enumerate(LABELS):
            if label in col_for_label:
                proba_aligned[j] = proba_row[col_for_label[label]]

        sarvam = self._sarvam.classify(message)
        out = ensemble_predict(proba_aligned, sarvam["intent"],
                               sarvam["confidence"], self.weight_bge)
        return {
            "intent": out["intent"],
            "confidence": round(out["confidence"], 2),
            "method": "v4_ensemble",
            "component_predictions": {
                "bge_lr": {"intent": LABELS[int(np.argmax(proba_aligned))],
                          "confidence": round(float(np.max(proba_aligned)), 2)},
                "sarvam": sarvam,
            },
        }
