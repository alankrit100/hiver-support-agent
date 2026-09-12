# Task: Build an AI Customer-Support Agent (Hiver SDE Intern Take-Home)

## Context

This is a take-home assignment. The grading criteria explicitly favors **rigorous, honest evaluation over a flashy system**. Your job is to build a working pipeline AND prove, with evidence, whether it actually works — including where it fails. Do not oversell results. Do not skip the self-critical sections. A mediocre agent with an honest, sharp evaluation is the goal — not a system that looks impressive but has hand-wavy proof.

I (the human) will review and personally do some steps myself — those are marked **[HUMAN]** below. Do not silently do these steps for me or fabricate outputs for them. When you reach a **[HUMAN]** step, stop, tell me exactly what you need from me, and wait.

---

## Stack & Tools

- **Language:** Python 3.11+
- **LLM:** Use Google Gemini (flash-tier model — check current model names/pricing, do not assume a name from training data; use whichever current Gemini Flash variant has a free tier as of today). Abstract the LLM call behind a single wrapper module (`llm_client.py`) so the model can be swapped later without touching pipeline logic.
- **Embeddings/retrieval:** Use a free/open embedding option (e.g., a local sentence-transformers model, or Gemini's embedding endpoint if free-tier covers it) for retrieving similar historically-resolved threads. Check current best lightweight options — don't assume.
- **Data:** Kaggle `thoughtvector/customer-support-on-twitter`. Use `kagglehub` or the Kaggle API for download. **Only use a subsample** — the assignment explicitly says the full dataset will not be run; keep it fast and reproducible (a fixed random seed, documented sample size).
- **Optional:** `PolyAI/banking77` from Hugging Face `datasets`, for intent naming inspiration only — do not treat it as ground truth for this dataset's intents.
- Use whatever current, actively-maintained libraries make sense (e.g., pandas, scikit-learn for baselines, a vector store like FAISS or chromadb for retrieval). Check latest stable versions — don't assume from training data.

---

## Phase 1 — Data Exploration & Brand Selection

1. Download a manageable subsample of the Kaggle dataset (document your sampling method — e.g., first N threads, or stratified by brand).
2. Analyze brand-level stats: message volume, average thread length (multi-turn depth), apparent resolution rate (does the brand actually reply and close out issues, or just template-deflect?).
3. Recommend the **top 2-3 brand candidates** with a one-paragraph rationale each (data richness, variety of issues, enough multi-turn "resolution" examples to ground replies on).
4. **[HUMAN]** — Present these candidates to me. I will pick the final brand. Do not proceed past this point until I confirm.

---

## Phase 2 — Intent Definition

1. Once the brand is chosen, sample ~150-300 real customer-initiated messages for that brand.
2. Cluster/skim them (via embeddings + clustering, or manual-style LLM categorization at scale) to surface naturally occurring issue types.
3. Propose 5-8 intent categories **grounded in what's actually in the data** (not copied wholesale from Banking77 — cross-reference for naming ideas only if useful).
4. **[HUMAN]** — Show me the proposed intents with 3-5 example messages each. I will sanity-check and approve/adjust before you hardcode them into the classifier. This is a judgment call I need to own and be able to defend.

---

## Phase 3 — Pipeline Build

Build these as separate, testable modules:

1. **Intent classifier** — LLM-based (few-shot with the approved intent definitions + examples). Include a confidence/uncertainty signal (e.g., ask the model to self-rate confidence, or use logprobs if available).
2. **Retrieval module** — given a new message + its intent, retrieve the most similar *historically resolved* threads from that brand (resolved = thread where the brand's agent replied and the conversation ended without further complaint — define and document your resolution heuristic).
3. **Reply drafter** — LLM call that drafts a reply grounded in the retrieved historical resolutions (use retrieval-augmented prompting; cite/reference which historical example(s) informed the draft, for traceability).
4. **Escalation decision** — combination of rules (e.g., low classifier confidence, keywords indicating anger/legal/safety/refund-amount thresholds, repeated unresolved turns) + an LLM-stated reason string. Must always output a human-readable reason, not just a boolean.

**Edge cases to explicitly handle and test:**
- Empty or very short messages ("??", "hi")
- Non-English or mixed-language messages
- Messages that are just @mentions/links with no real content
- Multi-issue messages (more than one intent in a single message)
- Sarcasm / all-caps anger — should likely trigger escalation, verify it does
- Duplicate/spam-like messages
- Threads with no clear resolution to retrieve from (cold-start case — define a sane fallback, e.g., generic brand-tone reply + auto-escalate)
- API failures / rate limits from the LLM/embedding provider — implement retries with backoff, and fail gracefully (log and skip, don't crash the pipeline)
- Cost/rate-limit ceilings on the free tier — batch and throttle calls, log token/request usage

Write this as a runnable CLI or notebook pipeline: input a raw message (or batch of messages) → output {intent, confidence, drafted reply, escalate: bool, escalation_reason}.

---

## Phase 4 — Golden Evaluation Set (150-250 examples)

**This step requires real human labeling. Do not let the LLM auto-generate ground-truth labels and pass them off as hand-labeled.**

1. Build a **labeling helper script** (simple CLI or a minimal local web UI) that:
   - Shows one message (+ thread context) at a time
   - Lets me type in: true intent, escalate (y/n), and optionally a short "ideal reply" note
   - Optionally shows an AI-suggested draft label **only after** I've entered my own answer first (to avoid anchoring bias) — this should be a toggle, defaulting to hidden-until-submitted.
   - Saves to a structured file (CSV/JSONL) with a timestamp and "labeled_by: human" field.
2. Sample the 150-250 examples with a documented method (e.g., stratified across your defined intents, with some deliberately hard/edge cases oversampled — document the ratio).
3. **[HUMAN]** — I will do the actual labeling using this tool, in two passes:
   - Pass A: ~20-30 examples labeled fully cold, no AI suggestion shown, to build genuine first-hand calibration.
   - Pass B: remaining examples with AI-suggested draft labels shown after my initial answer, which I then confirm or correct.
   - Do not proceed to build the eval harness metrics until I confirm this labeling is done and hand you the completed file.

---

## Phase 5 — Evaluation Harness

1. **Classification metrics:** precision/recall/F1 per intent, confusion matrix, against the golden set.
2. **Escalation metrics:** precision/recall against my golden escalate/no-escalate labels; also report false-negative escalations separately (missed escalations are worse than over-escalation — flag this asymmetry in the report).
3. **Reply quality — LLM-as-judge:**
   - Design a rubric (e.g., groundedness in retrieved history, relevance, correctness, tone-appropriateness, actionability) scored e.g. 1-5 per dimension.
   - **[HUMAN]** — I will personally score reply quality on a subset (e.g., 30-40 examples) using the same rubric, blind to the LLM judge's scores.
   - Compute agreement between my scores and the LLM judge's (e.g., Cohen's kappa or simple correlation). Report this plainly — including if agreement is mediocre. Do not cherry-pick a favorable subset.
4. **Baselines:**
   - Trivial baseline: e.g., single canned reply for everything / majority-class intent guess.
   - Simple baseline: e.g., TF-IDF + logistic regression classifier, template-based reply (no retrieval/LLM).
   - Report your full pipeline against both, honestly — including any cases where a baseline surprisingly does as well or better.

---

## Phase 6 — Report (README section, max 6 pages)

Draft this, but flag every section marked [HUMAN] for my input before finalizing:

- Problem framing: what "good" means for this brand + what you chose not to build. **[HUMAN: I need to personally write/edit this — it's a judgment call I must be able to defend live.]**
- Results vs. both baselines, with tables/plots.
- Failure analysis: top 5 failure modes with real examples + hypotheses. **[HUMAN: draft this, but I will personally read the failure examples and sanity-check/rewrite the hypotheses — I need to actually understand each one.]**
- "What is misleading about my headline number?" — mandatory section. **[HUMAN: I will write this section myself, or heavily edit your draft. This needs to be genuinely self-critical, not performative.]**
- What you'd do next with one more week.

---

## Phase 7 — Decision Log

Maintain a running `DECISIONS.md` throughout — 10-15 bullet points of non-obvious calls (e.g., resolution heuristic, intent boundaries, escalation thresholds, sampling method) with a one-line "why." **[HUMAN: I will review and be able to personally justify every line of this live in the interview — do not let this become a list of things only you understand.]**

---

## Deliverables Checklist

- [ ] `README.md` — setup + reproduce-in-15-minutes instructions
- [ ] Runnable pipeline code
- [ ] `golden_eval_set.jsonl` (human-labeled, per Phase 4)
- [ ] Evaluation harness code + results (metrics, LLM-judge, human-agreement numbers)
- [ ] Report (README section or separate doc, ≤6 pages)
- [ ] `DECISIONS.md`
- [ ] All code committed to a public (or access-granted private) repo

## Non-negotiables

- Do not fabricate or auto-generate the golden labels without human review.
- Do not skip the "misleading headline number" self-critique.
- Cite any borrowed code/prompts/snippets clearly (in comments or a CREDITS.md).
- Everything must run on a documented subsample in under 15 minutes, per the assignment's own constraint.
- Flag every [HUMAN] checkpoint clearly and wait for my input — do not assume and proceed.
