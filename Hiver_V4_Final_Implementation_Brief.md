# Hiver SDE Intern Take-Home
## V4 Implementation Brief for OpenCode

## Objective

Upgrade the current AppleSupport customer-support agent into a defensible V4 system for the Hiver take-home assignment.

Do not rewrite the project from scratch. Inspect the existing repository first, preserve working V1/V2/V3 experiments, and add V4 cleanly.

The priority is **evaluation quality and reproducibility**, not complexity.

---

# 1. Current Project Facts

### Brand
- Brand: `AppleSupport`
- Approximate source volume: 106K tweets for this brand.
- Dataset: Customer Support on Twitter.
- Existing pipeline reconstructs multi-turn context and identifies AppleSupport replies.

### Human-labelled dataset
215 labelled examples across 6 intents:

| Intent | Count |
|---|---:|
| unclear | 50 |
| ios_update_issues | 39 |
| account_access | 39 |
| battery_drain_after_update | 38 |
| app_malfunction | 32 |
| device_crash_freeze | 17 |

`device_crash_freeze` is the smallest class and must be watched carefully during evaluation.

### Existing final test set
- `data/test_set.jsonl`
- 45 examples.
- Originally created as a stratified 79/21 split from the 215 labelled examples.
- **This 45-example test set is now FROZEN. Do not tune V4 against it.**
- It must only be used once for final headline evaluation after all model selection is complete.

### Remaining development data
- 170 labelled examples.
- Use these for model development and cross-validation.

### Current LLM
- Sarvam API.
- Model: `sarvam-105b`.
- Existing client is in `src/llm_client.py`.

### Existing thread/retrieval data
The existing retrieval pipeline already produces data containing fields similar to:

- `tweet_id`
- `text`
- `thread_context`
- `brand_response`
- `labeled_intent`

Existing thread reconstruction and resolution heuristics should be preserved unless a concrete bug is found.

---

# 2. Lessons From V1-V3

Do not discard these experiments. Keep them available for comparison and report them accurately.

### V1: LLM + keyword fallback
Result: **24.7% accuracy (53/215)**.

Problems:
- LLM frequently defaulted to `unclear`.
- Keyword fallback was brittle.
- Edge-case heuristics added complexity without solving the main problem.

### V2: BGE-M3 + FAISS + LLM hybrid
Result: **55.6% accuracy** on the prior evaluation.

Problem discovered:

> Semantic similarity is not the same thing as intent similarity.

Tweets can be semantically similar while requiring different support actions/intents.

### V3: Sarvam few-shot LLM-only classifier
Result: **62.2% accuracy (28/45)** on the current 45-example test set.

It is the best of V1-V3, but the 45-example test set is too small for confident model selection, and V3 has been tuned against that test set over multiple iterations.

Do not claim 62.2% as the final generalization result for V4.

---

# 3. V4 Classification Goal

Build a real supervised intent classifier and compare it against the current Sarvam classifier.

## Required candidates

Implement and evaluate at least these approaches:

1. **TF-IDF + Logistic Regression**
2. **BGE-M3 embeddings + Logistic Regression**
3. **BGE-M3 embeddings + Linear SVM**
4. **Current Sarvam-105B few-shot classifier** as an LLM baseline

Do not assume which one will win.

## Development protocol

The 45-example test set must remain untouched.

On the 170-example development set:

- Use **5-fold stratified cross-validation**.
- Select model/hyperparameters using cross-validation only.
- Report mean and standard deviation where practical.
- After model selection, fit the winning supervised model on all 170 development examples.
- Evaluate it exactly once on the frozen 45-example test set.

Do NOT make another train/test split from the 170 data unless there is a specific technical reason that cannot be handled by cross-validation.

## Classification metrics

Report:
- Accuracy
- Macro F1
- Per-intent precision, recall, F1
- Confusion matrix
- Class support

Macro F1 is required because the classes are imbalanced.

## `unclear` intent

Do not automatically remove `unclear`.

First inspect examples and determine whether it represents a genuine support state such as insufficient/ambiguous customer information.

If it is a legitimate intent/category, keep it.
If it is merely a garbage bucket caused by labelling uncertainty, document that and recommend a taxonomy change rather than silently deleting it.

---

# 4. V4 Retrieval Design

Keep the existing FAISS/BGE-M3 infrastructure where useful, but improve retrieval for grounded reply generation.

### Retrieval requirements

For a customer message:

1. Predict intent.
2. Restrict candidate historical examples to the same brand.
3. Prefer the predicted intent.
4. Perform semantic similarity search within that constrained pool.
5. Retrieve top 3-5 useful historical cases.
6. Preserve enough context to understand how AppleSupport actually resolved the issue.

Do not rely on pure semantic nearest-neighbour search across all intents.

The goal is not to find textually similar tweets. The goal is to find **historically similar support situations and resolutions**.

Log retrieval quality during evaluation:
- retrieval hit count
- similarity scores
- whether retrieved examples share the gold intent when available
- whether a usable historical resolution was found

---

# 5. Reply Generation With Sarvam

Continue using `sarvam-105b` for reply generation.

The LLM should receive:
- customer message
- relevant thread context
- predicted intent
- top retrieved historical cases
- actual historical brand replies/resolutions where available
- explicit instruction to avoid inventing policies, refunds, timelines, links, or actions

The generated answer must be grounded in the retrieved evidence.

### Important change from current pipeline

**Do NOT use the confidence threshold to suppress reply drafting.**

Bad pattern:

`low confidence -> generic "I don't have enough information" reply`

Required pattern:

`low confidence -> still generate the best grounded draft possible -> mark for human escalation`

That way the human agent gets a useful draft instead of a dead-end fallback.

---

# 6. Escalation Logic

Escalation should be treated as a separate decision from reply generation.

The system should produce:

```text
intent
intent_confidence
reply
escalate
escalation_reason
retrieval_evidence
```

### Suggested escalation rules

Escalate when one or more of these apply:

- Low classifier confidence.
- Weak or missing retrieval evidence.
- No historical resolution that supports the draft.
- Sensitive/high-risk support situation.
- Conflicting historical resolutions.
- The model would need to invent information to answer safely.

Auto-handle when:

- Intent confidence is high.
- Evidence is strong and relevant.
- The requested support action is routine.
- The reply can be grounded in historical evidence without inventing facts.

Make the rules explicit in code and document them.

---

# 7. Reply Evaluation

The current judge results are weak:

- Historical baseline overall: 2.62/5
- Improved pipeline overall: 2.14/5

Current failure signal:
- Only about 5/40 pipeline replies had citations/evidence.
- Too many outputs became generic fallback responses.

V4 must address this.

Evaluate at least:

- Groundedness
- Relevance
- Correctness
- Tone
- Actionability

Use a 1-5 scale with a clearly documented rubric.

Also report the percentage of replies that:
- contain usable evidence/citations
- are unsupported by retrieval evidence
- trigger escalation
- use generic fallback text

---

# 8. LLM-as-Judge Validation

Hiver explicitly requires evidence that the LLM judge agrees with a human.

Create a small human-validation subset from the golden evaluation set.

Recommended:
- 30-50 examples
- Human scores using the exact same rubric as the LLM judge
- Compare human and LLM scores

Report at least one clear agreement statistic, for example:
- exact agreement rate
- average absolute score difference
- Spearman correlation if appropriate

Do not claim the LLM judge is trustworthy without this validation.

---

# 9. Final Golden Evaluation Set

The assignment requires 150-250 hand-labelled examples.

The existing 215 labelled examples are suitable as the golden set only if their labels and sampling methodology are documented clearly.

Document:
- how examples were sampled
- how threads were reconstructed
- how intents were defined
- who labelled them
- how ambiguous examples were handled

Most importantly, distinguish clearly between:

- **Development data:** 170 examples used for CV/model selection.
- **Frozen test set:** 45 examples used exactly once for final classification evaluation.
- **Golden evaluation set:** the full labelled set / appropriately documented evaluation collection.

Avoid any wording that implies the 45-example test set was repeatedly used for tuning.

---

# 10. Baselines

The final report must compare against at least two baselines.

Use:

### Baseline A: Majority-class classifier
Always predict the most frequent intent.

### Baseline B: Simple lexical classifier
TF-IDF + Logistic Regression is acceptable as the simple ML baseline.

Also report the Sarvam few-shot classifier as an LLM baseline.

For reply generation, keep a simple historical-retrieval/template baseline where practical.

---

# 11. Evaluation Leakage Rules

These are mandatory.

- Never train on the frozen 45-example test set.
- Never select hyperparameters using the 45-example test score.
- Never choose the architecture because it scored best on the 45 examples.
- Do not use test examples as few-shot demonstrations.
- Do not rewrite test labels after seeing predictions.
- All model selection happens on the 170-example development set using CV.
- Only after the model is frozen should the 45-example test evaluation be run.

If existing files contain accidental leakage, fix it and document the correction.

---

# 12. Reproducibility

The README must let a reviewer reproduce the headline results quickly.

Add a single command or very small sequence that:

1. Loads the labelled development/test data.
2. Runs the V4 CV/model selection.
3. Fits the selected classifier.
4. Runs final frozen-test evaluation.
5. Produces the metrics table and confusion matrix.

Avoid requiring the full 3M-tweet dataset for the evaluation run.

Do not commit API keys.

Document required environment variables, especially:

`SARVAM_API_KEY`

---

# 13. Files To Add Or Update

Keep the existing code structure where possible. Suggested additions:

```text
src/
  v4_classifier.py
  v4_evaluation.py
  retrieval.py              # update only if needed
  reply_generator.py        # update only if needed
  escalation.py             # update if needed

scripts/
  run_v4_cv.py
  run_v4_final.py
  evaluate_replies.py
  validate_llm_judge.py

reports/
  v4_results.json
  v4_confusion_matrix.png

README.md
DECISION_LOG.md
```

Do not create duplicate implementations if existing modules already provide the functionality cleanly.

---

# 14. Required Final Output Schema

Use a stable JSON structure similar to:

```json
{
  "intent": "battery_drain_after_update",
  "intent_confidence": 0.87,
  "reply": "...",
  "escalate": false,
  "escalation_reason": null,
  "retrieval_evidence": [
    {
      "tweet_id": "...",
      "similarity": 0.84,
      "historical_response": "..."
    }
  ]
}
```

Use `escalation_reason` whenever `escalate=true`.

---

# 15. Decision Logging

Add 10-15 non-obvious decisions to `DECISION_LOG.md`.

At minimum document:

1. Why AppleSupport was selected.
2. Why the 45-example test set is frozen.
3. Why 5-fold CV is used on the 170 development examples.
4. Why semantic nearest-neighbour voting was not retained as the final classifier.
5. Why supervised embeddings are being tested.
6. Why Macro F1 is emphasized.
7. Why `unclear` was retained or changed.
8. Why retrieval is constrained by predicted intent.
9. Why low confidence triggers escalation rather than no reply.
10. Why historical evidence is required for auto-handling.
11. Why Sarvam is used for generation rather than classification only / vice versa.
12. How the LLM judge was validated against humans.
13. Any major retrieval or resolution heuristic that affects the results.

---

# 16. Acceptance Criteria

V4 is complete only when all are true:

- [ ] Existing V1/V2/V3 remain reproducible/comparable.
- [ ] 45-example test set is frozen and untouched during tuning.
- [ ] 5-fold stratified CV is used on the 170 development examples.
- [ ] TF-IDF + Logistic Regression is implemented.
- [ ] BGE-M3 + Logistic Regression is implemented.
- [ ] BGE-M3 + Linear SVM is implemented.
- [ ] Sarvam few-shot remains as an LLM baseline.
- [ ] Final classifier is selected using development CV only.
- [ ] Final classifier is evaluated once on the frozen 45-example test set.
- [ ] Accuracy, Macro F1, per-intent F1 and confusion matrix are reported.
- [ ] Retrieval is constrained by brand and predicted intent.
- [ ] Reply generation uses historical evidence.
- [ ] Low-confidence cases can still receive a useful draft reply.
- [ ] Escalation has an explicit reason.
- [ ] Reply quality is evaluated on groundedness, relevance, correctness, tone and actionability.
- [ ] LLM judge is validated against human scores.
- [ ] Baselines are documented.
- [ ] Failure cases are saved with real examples.
- [ ] README reproduces headline results without the full 3M-row dataset.
- [ ] No API keys or secrets are committed.

---

# 17. Implementation Order

Follow this order exactly:

### Step 1
Inspect the entire existing repository and summarize:
- current files
- current data flow
- current evaluation flow
- where V1/V2/V3 live
- what can be reused

Do not edit code yet.

### Step 2
Inspect the 215 labelled examples and confirm:
- class distribution
- duplicate/near-duplicate risk
- whether `unclear` is meaningful
- whether the 45 test examples are correctly isolated

### Step 3
Implement the V4 classifier candidates and 5-fold CV.

### Step 4
Select the winner using development CV only.

### Step 5
Fit the winning classifier on all 170 development examples.

### Step 6
Run the frozen 45-example test exactly once.

### Step 7
Fix retrieval using predicted-intent constraints and measure retrieval quality.

### Step 8
Fix Sarvam reply generation so low confidence does not automatically produce useless fallback responses.

### Step 9
Implement explicit escalation policy and output reasons.

### Step 10
Run reply evaluation and human-vs-LLM judge validation.

### Step 11
Produce final tables, confusion matrix, failure examples and decision log.

### Step 12
Update README and report only after all numbers are final.

---

# 18. Important Constraint

Do not optimize for a single impressive number.

The final system should be defensible under questioning.

The report should explicitly answer:

> **“What is misleading about my headline number?”**

Examples of honest caveats may include:
- small frozen test set
- class imbalance
- limited number of hand-labelled examples
- judge limitations
- retrieval availability differences
- evaluation only on one brand
- historical data reflecting past support policy rather than current policy

Never hide these limitations.

---

# 19. What OpenCode Should NOT Do

- Do not blindly rewrite the whole project.
- Do not remove V1/V2/V3.
- Do not tune against the frozen 45-example test set.
- Do not use a more complicated model merely because it sounds more advanced.
- Do not claim a score is meaningful without checking the evaluation design.
- Do not fabricate historical evidence.
- Do not invent AppleSupport policies in generated replies.
- Do not convert every low-confidence case into an unhelpful fallback message.
- Do not hide poor results. Record and explain them.

The goal is a clean, reproducible, explainable V4 system that can survive a live technical interview.
