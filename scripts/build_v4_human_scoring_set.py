"""Regenerate the human-scoring set using the FIXED V4 pipeline.

The previous data/human_scoring_template_v2.json was generated before the
V4 fixes to reply_drafter.py/retrieval.py/escalation.py -- most of its
replies are the old generic fallback text, not real grounded drafts. This
script reuses the SAME 25 tweet_ids (for continuity) but regenerates the
reply for each one by running it through the current, fixed pipeline, then
scores each fresh reply with the LLM judge.

This produces a template ready for human_scorer.py to fill in blind (it
never shows the LLM score while scoring -- only for the final agreement
calc). Actual human scoring is NOT done here and must not be.

Usage (run from repo root):
    python scripts/build_v4_human_scoring_set.py

Outputs:
    data/human_scoring_template_v3.json / .csv   (human_score fields empty)
    reports/v4_llm_judge_results.json            (LLM judge scores + stats)
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

GOLDEN_PATH = REPO_ROOT / "data" / "golden_eval_set.jsonl"
V2_TEMPLATE_PATH = REPO_ROOT / "data" / "human_scoring_template_v2.json"
OUT_JSON_PATH = REPO_ROOT / "data" / "human_scoring_template_v3.json"
OUT_CSV_PATH = REPO_ROOT / "data" / "human_scoring_template_v3.csv"
JUDGE_RESULTS_PATH = REPO_ROOT / "reports" / "v4_llm_judge_results.json"

JUDGE_PROMPT = """Rate this customer support reply on 5 dimensions (1-5 scale):

1. Groundedness: Is it based on real examples? Does it cite specific cases?
2. Relevance: Does it address the actual issue?
3. Correctness: Is the advice accurate?
4. Tone: Professional and empathetic?
5. Actionability: Clear next steps?

Customer: {message}
Reply: {reply}

Respond ONLY in JSON:
{{"groundedness": <1-5>, "relevance": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>}}"""


def load_golden_by_id() -> dict:
    rows = {}
    with open(GOLDEN_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["tweet_id"]] = r
    return rows


def score_reply(llm, message: str, reply: str) -> dict:
    prompt = JUDGE_PROMPT.format(message=message[:200], reply=reply[:200])
    try:
        response = llm.generate(prompt)
        if response is None:
            return {"error": "LLM returned None"}
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": f"Could not parse: {response[:100]}"}
    except Exception as e:
        return {"error": str(e)}


def compute_statistics(scores: list) -> dict:
    dims = ["groundedness", "relevance", "correctness", "tone", "actionability"]
    stats = {}
    for dim in dims:
        vals = [s[dim] for s in scores if dim in s and isinstance(s[dim], (int, float))]
        stats[dim] = {
            "mean": round(sum(vals) / len(vals), 2) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "count": len(vals),
        }
    means = [stats[d]["mean"] for d in dims if stats[d]["count"] > 0]
    stats["overall"] = round(sum(means) / len(means), 2) if means else 0
    return stats


def main() -> None:
    from llm_client import LLMClient
    from pipeline import SupportPipeline

    v2 = json.loads(V2_TEMPLATE_PATH.read_text())
    tweet_ids = [e["tweet_id"] for e in v2]
    golden_by_id = load_golden_by_id()

    print(f"[BUILD-V3] Initializing fixed pipeline for {len(tweet_ids)} messages...")
    pipeline = SupportPipeline(mode="v4_ensemble")
    llm = LLMClient()

    entries = []
    t0 = time.time()
    for i, tid in enumerate(tweet_ids):
        golden = golden_by_id.get(tid)
        if golden is None:
            print(f"[BUILD-V3] WARNING: tweet_id {tid} not found in golden_eval_set.jsonl, skipping")
            continue
        message = golden["text"]

        result = pipeline.process_message(message)
        judge_score = score_reply(llm, message, result["reply"])
        judge_score["tweet_id"] = tid

        entries.append({
            "tweet_id": tid,
            "message": message,
            "intent": result["intent"],
            "reply": result["reply"],
            "escalate": result["escalate"],
            "reply_grounding": result["reply_grounding"],
            "llm_score": judge_score,
            "human_score": {"groundedness": None, "relevance": None,
                           "correctness": None, "tone": None, "actionability": None},
        })
        print(f"[BUILD-V3] {i + 1}/{len(tweet_ids)} done ({time.time() - t0:.0f}s elapsed)")

    OUT_JSON_PATH.write_text(json.dumps(entries, indent=2))

    with open(OUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet_id", "message", "intent", "reply", "escalate", "reply_grounding",
                         "groundedness", "relevance", "correctness", "tone", "actionability"])
        for e in entries:
            writer.writerow([e["tweet_id"], e["message"], e["intent"], e["reply"], e["escalate"],
                             e["reply_grounding"], "", "", "", "", ""])

    valid_scores = [e["llm_score"] for e in entries if "error" not in e["llm_score"]]
    stats = compute_statistics(valid_scores)
    JUDGE_RESULTS_PATH.parent.mkdir(exist_ok=True)
    JUDGE_RESULTS_PATH.write_text(json.dumps({
        "scores": [e["llm_score"] for e in entries],
        "statistics": stats,
        "sample_size": len(entries),
        "approach": "LLM-as-Judge on V4-fixed pipeline replies (post retrieval/drafter/escalation fixes)",
        "fallback_reply_count": sum(1 for e in entries if e["reply_grounding"] == "fallback_cold_start"),
    }, indent=2))

    print("\n" + "=" * 70)
    print(f"Built {len(entries)} fresh entries -> {OUT_JSON_PATH.name} / {OUT_CSV_PATH.name}")
    print(f"Fallback (cold-start) replies: "
          f"{sum(1 for e in entries if e['reply_grounding'] == 'fallback_cold_start')}/{len(entries)} "
          f"(old broken pipeline had ~{sum(1 for e in v2 if 'help you with this' in e.get('reply', ''))}/25)")
    for dim, s in stats.items():
        if dim != "overall":
            print(f"  {dim:15s} mean={s['mean']:.2f} (n={s['count']})")
    print(f"  overall mean: {stats['overall']}")
    print(f"\nJudge results: {JUDGE_RESULTS_PATH}")
    print(f"\nNext: python src/human_scorer.py --template {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
