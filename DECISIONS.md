# DECISIONS.md

Non-obvious decisions with one-line rationale.

---

## Data & Sampling

1. **Brand: AppleSupport** — Rich data (106K tweets), multi-turn threads, clear resolution signals.

2. **Sample size: 215 messages** — Balances statistical significance with LLM API costs (~₹77 budget remaining).

3. **Stratified sampling** — Ensures all 6 intents are represented proportionally.

4. **Train/test split: 170/45** — 80/20 split with stratification to maintain class balance.

---

## Intent Definition

5. **6 intents (not 5 or 8)** — `unclear` added as 6th intent to handle ambiguous messages (reduces misclassification into other categories).

6. **Intent boundaries are judgment calls** — e.g., "apps crash" could be `app_malfunction` or `device_crash_freeze`. Documented in `intent_definitions.json` with boundary notes.

---

## Classification

7. **LLM-only (no regex/keyword/fuzzy)** — Hybrid approach failed (-6.6% accuracy). Semantic similarity doesn't correlate with intent similarity. **[SUPERSEDED by V4, decision #16-18]** — the system is no longer LLM-only; the deployed classifier is `bge_lr` (a supervised model) ensembled with the Sarvam few-shot classifier, selected via CV specifically because pure LLM-only underperformed it.

8. **Removed chain-of-thought reasoning** — Sarvam AI returns `None` on longer prompts. Removing reasoning saves ~27% cost with minimal accuracy loss.

---

## Retrieval

9. **FAISS with MiniLM-L6-v2** — Fast (384-dim), good enough for Twitter support. BGE-M3 (1024-dim) tested but 2-3x slower with marginal improvement.

10. **top_k=3 for retrieval** — More examples dilute the prompt. **[PARTIALLY SUPERSEDED]** The original similarity>0.4 filter this decision described is gone from `retrieval.py` after the V4 intent-constrained rewrite (decision #16-ish territory, see the retrieval section of the code) — it now returns the top_k intent-preferred matches regardless of absolute similarity, with weak evidence caught downstream by escalation's `WEAK_EVIDENCE_SIMILARITY = 0.4` check instead of being filtered out of retrieval itself.

---

## Reply Generation

11. **Citation enforcement in prompt** — requiring the reply to name "Example 1/2/3" explicitly, not vague phrasing like "based on similar cases." This was weak pre-V4 (only 12.5% of replies actually cited an example). After the V4 reply-drafter fix (confidence no longer suppresses drafting, so the LLM is actually asked to cite far more often), 25/25 replies in the current human-scoring set name a specific example — but see [What is Misleading About My Headline Number?](#what-is-misleading-about-my-headline-number): citing an example isn't the same as the LLM judge agreeing the reply is genuinely grounded in it.

12. **Intent-specific guidelines** — Different intents need different reply structures (e.g., `account_access` vs `device_crash_freeze`).

13. **Confidence threshold: 0.3** — Below this, return fallback message. Lowered from 0.3 to allow more LLM-generated replies. **[SUPERSEDED — this pattern was explicitly removed.]** `reply_drafter.py` no longer uses classifier confidence to gate drafting at all; it always attempts a grounded draft when retrieval evidence exists, and confidence only affects the separate escalation decision (see `escalation.py`). This was the anti-pattern the V4 fixes specifically targeted — low confidence was producing generic fallback replies even when good retrieval evidence existed.

---

## Evaluation

14. **LLM-as-Judge on pipeline output** — Historical replies scored lower because they can't cite retrieval examples. Must score pipeline-generated replies for fair measurement.

15. **Simple baseline is overfitted** — 95.8% accuracy but trained/tested on same data. Reported honestly as "not production-ready."

---

## V4: Classifier Ensemble

16. **Wired the winning V4 classifier (bge_lr) into the actual pipeline** — previously `pipeline.py` still ran the old Sarvam-only/hybrid classifiers while the validated bge_lr model only existed inside the CV/eval scripts. `--mode v4_ensemble` is now the default.

17. **Ensembled bge_lr with the Sarvam few-shot classifier (weighted vote)** — the two are trained/prompted completely differently and don't make the same mistakes; a blend was cheap to try and had a clear evaluation protocol already in place. See `src/ensemble_classifier.py`, `scripts/run_v4_ensemble_cv.py`, `scripts/run_v4_ensemble_final.py`.

18. **Sarvam's single prediction is converted to a pseudo-distribution, not a real probability** — Sarvam only outputs one intent + a self-reported confidence, not per-class probabilities. I put `confidence` mass on the predicted label and split the rest uniformly over the other 5. This is a heuristic (documented in `ensemble_classifier.py`), not a calibrated distribution — don't over-interpret the blended "confidence" as a true probability.

19. **Blend weight (weight_bge=0.8) selected by 5-fold CV on dev-170 only** — same leakage discipline as the original V4 classifier selection; the frozen 45-example test set was never touched during weight selection, only for the final one-shot evaluation.

20. **Frozen test-45 result: ensemble reached 68.9% accuracy / 0.691 macro F1, vs 57.8%/0.541 for bge_lr alone** — a real gain, but from a single 45-example run; treat the exact gap as noisy, not a precise measurement. 2 of the 45 Sarvam calls failed transiently and fell back to `unclear`/0.0 confidence during this run — negligible at this scale, but noted for honesty since the eval protocol calls for exactly one clean run.

21. **Removed "still"/"keeps"/"again"/"continues" from the repeated-frustration escalation keywords** — these fire on ordinary first-contact bug descriptions ("phone keeps shutting off" describes a recurring device symptom, not a repeated support contact), a false positive documented in `PROGRESS.md` from early testing. Kept only phrases that specifically imply "I've contacted you about this before" (e.g. "already told", "multiple times").

22. **Switched the human-scoring-set generation from Sarvam to Groq (`openai/gpt-oss-120b`)** — `sarvam-105b` returned empty content on ~40-45% of drafting calls in testing (a documented reasoning-model quirk: it can burn its whole output budget on internal thinking on longer prompts). Groq was stress-tested (6/6 clean runs on the same prompt shape) and produced 25/25 clean replies with 0 fallbacks, vs. Sarvam's 11/25 fallback rate on the same 25 messages. `llm_client.py` auto-detects the provider from which API key is set, so this required zero pipeline code changes elsewhere -- exactly the "swap without touching pipeline logic" design from `instructions.md`.

23. **Did NOT re-run the frozen V4 classification numbers with Groq** — those (57.8%/68.9% on the frozen test-45) were already validated once under the leakage protocol and must never be re-run after the fact. The provider swap only applies to reply-drafting/judging going forward, not the classification headline numbers.

24. **Pinned `scripts/run_v4_cv.py` and `scripts/run_v4_ensemble_final.py` to `provider="sarvam"` explicitly** — after adding `GROQ_API_KEY`, `LLMClient()`'s auto-detect would silently switch these frozen, already-reported baseline/ensemble numbers to Groq on any re-run, breaking reproducibility of the 68.9%/0.691 result. Fixed by constructing `LLMClient(provider="sarvam")` explicitly at those two call sites only; the production pipeline (`ensemble_classifier.py`) is left on auto-detect since it should prefer the more reliable backend.

## Key Learnings

- **Hybrid failed:** Semantic similarity ≠ intent similarity
- **Groundedness is hard:** Requires explicit citation enforcement, and even then an LLM judge and a human can disagree on what counts as "grounded enough"
- **Confidence should drive escalation, not drafting:** Gating replies on confidence just produces useless fallback text even when good evidence exists (superseded, see #13, #17)
- **External APIs fail in ways that quietly change your numbers:** a credit outage or an empty-response quirk can look like a normal result unless you're checking for it
- **Honest evaluation > high scores:** the assignment favors self-critique over a flashy number
