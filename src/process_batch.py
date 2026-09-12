"""Process messages in batches with progress saving."""

import json
import time
from pathlib import Path

# Load sample
sample_path = Path(__file__).parent.parent / "data" / "judge_sample.json"
with open(sample_path) as f:
    sample = json.load(f)

print(f"Processing {len(sample)} messages in batches...")

# Import and initialize pipeline
from pipeline import SupportPipeline
pipeline = SupportPipeline()

# Process in batches of 5
batch_size = 5
all_results = []

for batch_start in range(0, len(sample), batch_size):
    batch_end = min(batch_start + batch_size, len(sample))
    batch = sample[batch_start:batch_end]
    
    print(f"\nProcessing batch {batch_start//batch_size + 1} (messages {batch_start+1}-{batch_end})...")
    
    for i, entry in enumerate(batch):
        try:
            result = pipeline.process_message(entry["text"])
            all_results.append({
                "tweet_id": entry["tweet_id"],
                "text": entry["text"],
                "labeled_intent": entry["labeled_intent"],
                "brand_response": entry.get("brand_response", ""),
                "generated_reply": result.get("drafted_reply", ""),
                "detected_intent": result.get("intent", ""),
                "confidence": result.get("confidence", 0),
                "citations": result.get("reply_citations", []),
                "grounding": result.get("reply_grounding", ""),
            })
            print(f"  {batch_start + i + 1}. Tweet {entry['tweet_id']} (conf: {result.get('confidence', 0):.2f})")
        except Exception as e:
            all_results.append({
                "tweet_id": entry["tweet_id"],
                "text": entry["text"],
                "labeled_intent": entry["labeled_intent"],
                "brand_response": entry.get("brand_response", ""),
                "error": str(e),
            })
            print(f"  {batch_start + i + 1}. Error: {e}")
        
        time.sleep(0.3)
    
    # Save progress after each batch
    progress_path = Path(__file__).parent.parent / "data" / "pipeline_rerun_results_v2.json"
    with open(progress_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"  Saved {len(all_results)} results so far")

# Final save
print(f"\nDone! Total results: {len(all_results)}")

# Check citations
replies_with_citations = [r for r in all_results if r.get("citations")]
print(f"Replies with citations: {len(replies_with_citations)}")
