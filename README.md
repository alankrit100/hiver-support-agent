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

I picked AppleSupport because it's the only brand in the dataset with real multi-turn threads and visible resolutions — without that, grounded replies are impossible to even evaluate. "Good" for me meant three separable things:
- **Right intent** — the message gets classified into one of the 6 categories I defined from the data
- **A reply citing a real past resolution** — not a plausible-sounding invention, an actual historical example
- **Knowing when to escalate** — and saying why, in a human-readable reason

I decomposed it this way because it lets me measure each part independently instead of judging the system as one opaque black box.

I deliberately didn't build:
- **A fine-tuned model** — 170 labeled dev examples can't support it; a supervised classifier on top of pretrained embeddings was the honest ceiling for this data size
- **A multi-agent system** — one pipeline with clear, inspectable stages beats a set of agents I can't cleanly evaluate against each other
- **Real-time learning** — there's no feedback signal in this dataset to learn from; it would be simulating a capability I have no way to validate

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
| **LLM-Only (mine)** | **60.5%** | Best honest result, pre-V4 |

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

---

## What is Misleading About My Headline Number?

The 68.9% accuracy number sounds solid. It isn't as strong as it looks:

1. **The test set is only 45 messages.** Two or three examples going the other way and the number moves 4-7 points. This isn't a precise measurement, it's a rough one.
2. **Even my "clean" run wasn't fully clean.** 2 of the 45 predictions still failed silently mid-run. One got lucky and was right anyway, the other wasn't. So the number I'm reporting is already a couple points shakier than it looks on paper.
3. **Two different AI models are behind these results, not one.** One model classifies the message, a different one writes and judges the reply. It's not a single consistent system end to end, and I'd rather say that up front than let it look tidier than it is.
4. **This number describes the classifier, not what a customer gets.** About 44% of messages get sent to a human instead of auto-answered. So "68.9% accurate" doesn't mean "68.9% of customers get a good reply automatically" — it's one input into a bigger decision, not the whole outcome.

Bottom line: it's a real improvement over the earlier baselines, but it's a small, somewhat fragile number, not a settled fact.

---

## What I'd Do Next

With one more week:

1. **More labeled data.** 170 dev examples is the real ceiling on classification accuracy — more data would likely help more than further model tuning.
2. **Re-validate the LLM judge with a more differentiated human sample.** Redo the blind scoring pass deliberately trying to spread across the 1-5 range where quality genuinely differs, to get a statistically meaningful agreement number instead of a near-ceiling one.
3. **Decouple backend choice from headline numbers.** Run classification and reply generation on the same LLM backend for one clean, internally consistent report, and add automatic backend failover (Sarvam → Groq) so a single provider outage can't silently degrade results the way it did during development.
4. **Grow the retrieval pool** beyond 100 threads — right now over half of it is `unclear`, leaving only ~4 examples each for the smaller intents like `device_crash_freeze` and `app_malfunction`. Barely enough to search within.
5. **Recheck the escalation threshold.** The 0.4 confidence cutoff is inherited from an earlier version of the system and was never re-tuned for how the current ensemble's confidence scores actually spread out (0.21-0.70 on the test set). It's probably doing more work than it should.
6. **Deploy monitoring:** track real-world classifier confidence, escalation rate, and reply-groundedness over time rather than relying on one-shot evaluation snapshots.

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

