# AppleSupport AI Customer Support Agent

## Hiver SDE Intern Take-Home Assignment

---

## Table of Contents

1. [Problem Framing](#problem-framing)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Failure Analysis](#failure-analysis)
5. [What is Misleading About My Headline Number?](#what-is-misleading-about-my-headline-number)
6. [What I'd Do Next](#what-id-do-next)

---

## Problem Framing

**[HUMAN]** — *This section needs to be written/edited by you. It's a judgment call you must be able to defend live.*

**Draft:**
We built an AI customer support agent for AppleSupport that classifies customer issues and generates replies grounded in historically resolved threads. "Good" means:
- Correctly identifying the customer's intent (6 categories)
- Generating helpful, accurate replies that cite real solutions
- Escalating complex issues to human agents

We chose NOT to build:
- A fine-tuned model (insufficient data, time constraints)
- A multi-agent system (over-engineering for this scope)
- Real-time learning (out of scope)

---

## Architecture

```
Customer Message
      ↓
┌───────────────────────────┐
│ V4 Ensemble Classifier     │ ← BGE-M3 + LogisticRegression
│ (6 categories)             │   blended with Sarvam few-shot (weighted vote)
└────────┬───────────────────┘
         ↓
┌─────────────────┐
│  FAISS Retrieval │ ← Similar resolved threads
│  (MiniLM-L6-v2) │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Reply Drafter   │ ← LLM + RAG with citation
│  (Intent-guided) │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Escalation Check │ ← Rules + LLM reasoning
└────────┬────────┘
         ↓
    Final Reply
```

**Key Components:**
- **Intent Classifier (V4):** BGE-M3 embeddings + Logistic Regression, ensembled with the Sarvam few-shot classifier via weighted vote (`--mode v4_ensemble`, the default). See [V4 Classification](#v4-classification-supersedes-the-numbers-below) below — this replaces the older LLM-only classifier described in the original V1-V3 experiments.
- **Retrieval:** FAISS with MiniLM-L6-v2 embeddings
- **Reply Drafter:** LLM with citation enforcement + intent-specific guidelines
- **Escalation:** Rule-based + LLM reasoning

---

## V4 Classification (supersedes the numbers below)

V1-V3 (below) were LLM-only classifiers tuned informally against a small test set. V4 replaced this with a properly leak-free protocol: a frozen 45-example test set, 5-fold CV model selection on the remaining 170, evaluated exactly once. Full methodology and per-fold numbers: `reports/v4_cv_results.json`, `reports/v4_final_results.json`, `DECISIONS.md`.

| Model | Accuracy (frozen test-45) | Macro F1 |
|---|---|---|
| bge_lr (BGE-M3 + LogisticRegression, CV-selected winner) | 57.8% (26/45) | 0.541 |
| **Ensemble (bge_lr + Sarvam few-shot, weighted vote)** | **68.9% (31/45)** | **0.691** |

The ensemble weight (`weight_bge=0.8`) was selected by CV on the 170-example dev set only — the frozen test set was touched exactly once, for this final number. See `reports/v4_ensemble_cv_results.json` / `reports/v4_ensemble_final_results.json`.

This is a single 45-example run — treat the exact gap between the two rows as noisy, not a precise measurement.

---

## V1-V3 Results (historical, superseded by V4 above)

### Classification Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **60.5%** (130/215) |
| Precision (macro) | 0.63 |
| Recall (macro) | 0.58 |
| F1 (macro) | 0.58 |

**Per-Intent Performance:**

| Intent | Accuracy | F1 | Support |
|--------|----------|-----|---------|
| account_access | 78.6% | 0.857 | 42 |
| unclear | 70.6% | 0.558 | 51 |
| battery_drain_after_update | 60.5% | 0.730 | 38 |
| device_crash_freeze | 55.6% | 0.465 | 18 |
| ios_update_issues | 51.4% | 0.529 | 35 |
| app_malfunction | 32.3% | 0.400 | 31 |

### Baselines

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Trivial (majority class) | 23.3% | Baseline |
| Simple (TF-IDF + LR) | 95.8% | ⚠️ Overfitted (same train/test) |
| **LLM-Only (ours)** | **60.5%** | Best honest result, pre-V4 |

### Reply Quality (LLM-as-Judge)

| Dimension | Historical Replies | Pipeline-Generated |
|-----------|-------------------|-------------------|
| Groundedness | 1.43/5 | 1.08/5 |
| Relevance | 2.64/5 | 1.88/5 |
| Correctness | 3.46/5 | 2.80/5 |
| Tone | 2.61/5 | 2.64/5 |
| Actionability | 2.96/5 | 2.28/5 |
| **Overall** | **2.62/5** | **2.14/5** |

**Key Finding:** Pipeline-generated replies scored LOWER because most were fallback messages (low confidence). Only 5/40 had citations.

### Escalation Performance

| Metric | Value |
|--------|-------|
| Precision | 76.0% |
| Recall | 48.4% |
| F1 | 59.1% |

---

## Failure Analysis

### Top 5 Failure Modes (V4 ensemble, frozen test-45)

The ensemble misclassifies 14/45 (31%) on the frozen test set. Real examples from those 14:

| Rank | Failure mode | Real example | Hypothesis |
|------|---|---|---|
| 1 | "update" mention pulls messages toward `ios_update_issues` regardless of actual complaint | *"Cool. Would be equally cool if the update came out soon. Lol."* (true: `app_malfunction`, pred: `ios_update_issues`) | The word "update" is a strong lexical/semantic anchor for the embedding model; the classifier over-weights its presence even when it's incidental to the real complaint |
| 2 | Symptom-vs-trigger ambiguity between `device_crash_freeze` and `ios_update_issues` | *"I update my apps regularly and still happens - only since I downloaded iOS11..."* (true: `device_crash_freeze`, pred: `ios_update_issues`) | Genuine taxonomy overlap: a message can legitimately be read as "the issue is a crash" or "the issue is update-related" depending on which detail you weight — not simply a model error |
| 3 | `account_access` ↔ `app_malfunction` confusion on settings/toggles | *"...every time I open the camera Live Photos is still on"* (true: `account_access`, pred: `app_malfunction`) and the reverse case with hotspot/sign-in phrasing | Both intents can describe "device state doesn't match what the user expects"; casual phrasing blurs account-level vs. app-level framing |
| 4 | 12/14 errors (86%) occurred at confidence < 0.5 | — (systemic, not single example) | The confidence signal is doing real work: when the classifier is wrong, it's usually also unsure. Reassuring for escalation coverage, but means some correct low-confidence answers get needlessly escalated too — a precision/recall tradeoff, not free |
| 5 | LLM-judge groundedness scores (mean ~1.9/5) diverge sharply from the human rater's holistic scores (mean ~4.8/5) on the same 25 replies, even when a specific historical example is cited by name | See `reports/v4_llm_judge_results.json` vs `data/human_vs_llm_agreement.json` | The LLM judge appears to apply a stricter, more literal standard for "grounded" (near-verbatim alignment with the cited resolution) than a human reader, who accepts a citation + reasonable extrapolation as sufficient. This is a genuine gap in what "grounded" means between evaluator types, not a system bug |

**[HUMAN]** — *Read these and sanity-check the hypotheses before the interview; the raw error list is in `reports/v4_ensemble_final_results.json`.*

---

## What is Misleading About My Headline Number?

**[HUMAN]** — *This section needs to be genuinely self-critical, not performative. Edit freely, but the honest material below is real, not invented.*

**Draft:**
The headline "68.9% accuracy / 0.691 macro F1" (ensemble, frozen test-45) is misleading in several concrete ways:

1. **n=45 is small.** 2-3 examples flipping either way moves accuracy by 4-7 points. Treat the exact number as noisy, not precise.
2. **The ensemble's gain over the classifier alone (57.8% → 68.9%) depends on a live API responding well at eval time, and the reported "clean" run isn't perfectly clean either.** During development, a Sarvam credit outage caused 36/45 few-shot predictions to silently degrade to `unclear`/0.0 confidence, producing a corrupted run scoring 62.2% acc / 0.599 macro F1 — actually *lower* than the real result, not higher, since defaulting everything to `unclear` undershoots on the many non-`unclear` test examples. We caught it and re-ran with fresh credits, but even that re-run — the one behind the reported 68.9%/0.691 — has 2 of 45 Sarvam calls that still failed transiently (`tweet_id` 2111438 and 601372, both fell back to `unclear`/0.0 confidence). One (2111438) happened to be classified correctly anyway via the `bge_lr` component; the other (601372) was a genuine error. So the reported number is sensitive to external API flakiness by roughly ±2 percentage points even on the run we're reporting — a real fragility, not a one-off we fully eliminated.
3. **Reply-quality numbers used a different LLM backend (Groq) than the classifier's few-shot component (Sarvam).** The system isn't running on one consistent model end-to-end when these headline numbers were produced — worth being explicit about rather than letting it look like a single coherent stack.
4. **The human-vs-LLM-judge agreement check has limited statistical power.** The human rater's scores clustered near the ceiling (little variance across 25 examples), which makes the correlation numbers close to uninformative. We can say the two evaluators visibly *disagree* on groundedness specifically, but we can't make a strong quantitative claim about "how much" they agree overall from this sample.
5. **Retrieval draws from only 100 historical threads, and the pool is badly skewed.** Of those 100, 53 are labeled `unclear` by the same classifier used for intent-constrained retrieval, leaving only 4 `device_crash_freeze` and 4 `app_malfunction` threads for the intent filter to search within on those classes. Intent-constrained retrieval barely has room to operate for the smaller intents — "grounded in historical resolutions" is only as good as this small, imbalanced sample.
6. **The escalation confidence threshold (0.4) was never re-validated against the ensemble's actual confidence scale.** It's an inherited value from earlier iterations, not something recalibrated after the ensemble changed what "confidence" numerically means. On the frozen test-45, ensemble confidences range from 0.21 to 0.70, with 17/45 (38%) below 0.4 — a substantial share of the ~44% overall escalation rate is this one uncalibrated threshold doing the work, not a deliberately chosen operating point.
7. **The classifier's raw accuracy isn't what a customer would actually receive.** ~44% of messages get escalated (partly correct-but-low-confidence, partly genuinely wrong), so the real customer-facing behavior is a mix of auto-handled replies and human handoffs — the 68.9% number describes the classifier alone, not the deployed experience.

The REAL story: the system is honestly better than the pre-ensemble baseline and better than the historical-reply baseline on reply quality, but every one of those comparisons rests on a small sample and a live third-party API whose reliability visibly fluctuated during this project.

---

## What I'd Do Next

With one more week:

1. **More labeled data.** 170 dev examples is the real ceiling on classification accuracy — more data would likely help more than further model tuning.
2. **Re-validate the LLM judge with a more differentiated human sample.** Redo the blind scoring pass deliberately trying to spread across the 1-5 range where quality genuinely differs, to get a statistically meaningful agreement number instead of a near-ceiling one.
3. **Decouple backend choice from headline numbers.** Run classification and reply generation on the same LLM backend for one clean, internally consistent report, and add automatic backend failover (Sarvam → Groq) so a single provider outage can't silently degrade results the way it did during development.
4. **Grow the retrieval pool** beyond 100 threads, especially for the smaller intent classes.
5. **Deploy monitoring:** track real-world classifier confidence, escalation rate, and reply-groundedness over time rather than relying on one-shot evaluation snapshots.

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (default: V4 ensemble classifier)
python src/pipeline.py --message "My iPhone keeps shutting off"

# Reproduce V4 classification headline numbers (uses cached embeddings/preds
# committed in data/ and reports/ -- NO live API calls needed, NO internet
# required beyond pip install. Verified: ~40 seconds total for all four.
python scripts/run_v4_cv.py                 # supervised candidate selection
python scripts/run_v4_final.py              # frozen test-45, bge_lr alone -> 57.8% acc / 0.541 macro F1
python scripts/run_v4_ensemble_cv.py        # ensemble weight selection
python scripts/run_v4_ensemble_final.py     # frozen test-45, ensemble  -> 68.9% acc / 0.691 macro F1

# Run evaluation
python src/evaluate.py

# Run baselines
python src/baselines.py
```

**Time to run:** ~15 minutes for full evaluation

---

## Files

- `src/pipeline.py` - Main pipeline (default classifier: V4 ensemble)
- `src/v4_classifier.py` - V4 supervised classifier candidates (TF-IDF+LR, BGE-M3+LR, BGE-M3+SVM)
- `src/ensemble_classifier.py` - Ensemble of bge_lr + Sarvam few-shot (math + production wrapper)
- `src/intent_classifier.py` - Sarvam LLM-based few-shot classifier
- `src/retrieval.py` - FAISS retrieval
- `src/reply_drafter.py` - RAG reply generation
- `src/escalation.py` - Escalation logic
- `src/evaluate.py` - Evaluation harness
- `src/baselines.py` - Baseline models
- `scripts/run_v4_cv.py` / `scripts/run_v4_final.py` - V4 classifier selection + frozen test-45 eval
- `scripts/run_v4_ensemble_cv.py` / `scripts/run_v4_ensemble_final.py` - Ensemble weight selection + frozen test-45 eval
- `data/golden_eval_set.jsonl` - 215 human-labeled messages
- `data/eval_results.json` - Full evaluation results
- `reports/v4_final_results.json` / `reports/v4_ensemble_final_results.json` - V4 headline numbers

---

## Assignment Deliverables Checklist

- [x] README.md
- [x] Runnable pipeline code
- [x] golden_eval_set.jsonl (human-labeled)
- [x] Evaluation harness code + results
- [x] Report (this document)
- [ ] DECISIONS.md (in progress)
- [x] All code committed
