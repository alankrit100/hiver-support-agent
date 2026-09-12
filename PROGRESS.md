# Apple Support Agent - Progress Summary

## Completed

### Phase 1: Data Exploration
- Selected **AppleSupport** brand (106K tweets, 33K English responses, 23K unique customers)
- Downloaded full dataset: `data/twcs.csv` (2.8M rows, 493MB)
- Created 300-message sample: `data/apple_support_sample.csv`

### Phase 2: Intent Definition
- Defined **5 intents** with examples, boundary notes, and resolution heuristics:
  1. `ios_update_issues` - iOS/software update problems
  2. `battery_drain_after_update` - Battery performance issues
  3. `device_crash_freeze` - Device crashes, freezes, unexpected shutdowns
  4. `app_malfunction` - App crashes and malfunctions
  5. `account_access` - iCloud/Apple ID login issues
- Created `src/intent_definitions.json` with full intent specs

### Phase 3-4: Core Pipeline (Working)
- **Intent Classifier** (`src/intent_classifier.py`): LLM-based few-shot classification, tested and working
- **Retrieval Module** (`src/retrieval.py`): FAISS index with 100 resolved threads, <1s search
- **Reply Drafter** (`src/reply_drafter.py`): RAG-based generation with citations
- **Escalation Decider** (`src/escalation.py`): Rule-based with LLM fallback
- **Pipeline** (`src/pipeline.py`): CLI with `--message`, `--file`, `--interactive`, `--build-index`

### Phase 5: Testing
- Full pipeline tested with 6+ messages
- Intent classification working (85-100% confidence)
- Reply generation clean (stripped thinking process)
- Escalation logic functional (may need tuning for false positives)

## Current State

### Working
```bash
# Single message
python src/pipeline.py --message "My phone keeps shutting off"

# Interactive mode
python src/pipeline.py --interactive

# Build index (slow, use --max-threads to limit)
python src/pipeline.py --build-index
```

### Known Issues
1. **Escalation false positives**: "keeps" triggers repeated_frustration escalation
2. **LLM returning None**: Some prompts cause Sarvam AI to return None; handled gracefully with fallback
3. **Index build speed**: Full 36K thread index takes ~100 min; use `max_threads=100` for testing

## Files Created/Modified
- `requirements.txt` - Updated with sarvamai
- `.env.example` - Updated with SARVAM_API_KEY
- `src/llm_client.py` - Sarvam AI wrapper with retry + thinking stripping
- `src/intent_definitions.json` - 5 intents with examples/heuristics
- `src/intent_classifier.py` - LLM-based classifier
- `src/retrieval.py` - FAISS retrieval with early exit optimization
- `src/reply_drafter.py` - RAG reply generation
- `src/escalation.py` - Rule-based escalation
- `src/pipeline.py` - CLI entry point
- `analysis/brand_analysis.md` - Phase 1+2 analysis

## Next Steps (Optional)
1. Tune escalation rules to reduce false positives on "keeps"
2. Build larger FAISS index (500-1000 threads) for better retrieval
3. Add batch processing from JSONL file
4. Run evaluation metrics on test set
