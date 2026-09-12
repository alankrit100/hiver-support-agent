"""Sample evaluation candidates from twcs.csv for golden set labeling.

Usage:
    python src/sample_eval_set.py                    # Default: 200 messages
    python src/sample_eval_set.py --num-samples 150  # Custom count
    python src/sample_eval_set.py --seed 42          # Reproducible sampling
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Intent keyword matching for pre-bucketing (avoids LLM during sampling)
INTENT_KEYWORDS = {
    "ios_update_issues": [
        "update", "upgrade", "ios", "version", "install", "download",
        "software", "downloading", "updated", "downgrade",
    ],
    "battery_drain_after_update": [
        "battery", "drain", "charge", "charging", "power", "percent",
        "charger", "draining", "dead", "hot",
    ],
    "device_crash_freeze": [
        "crash", "freeze", "frozen", "hang", "shutdown", "shut off",
        "shutting", "unresponsive", "stuck", "boot", "restart",
        "sluggish", "slow", "laggy", "lag", "bricked",
    ],
    "app_malfunction": [
        "app", "message", "safari", "music", "itunes", "bluetooth",
        "setting", "audio", "sound", "call", "hear", "mic",
        "camera", "photo", "keyboard", "screen", "touch",
    ],
    "account_access": [
        "account", "id", "password", "icloud", "backup", "login",
        "sign in", "signin", "locked", "activation", "apple id",
    ],
}

# Edge case detection patterns
EDGE_CASES = {
    "short_message": lambda t: len(t.strip()) < 15,
    "non_english": lambda t: sum(1 for c in t if ord(c) > 127) > len(t) * 0.3,
    "mentions_only": lambda t: all(w.startswith("@") for w in t.split() if w),
    "links_only": lambda t: all(w.startswith("http") for w in t.split() if len(w) > 5),
    "all_caps": lambda t: len(t) > 20 and sum(1 for c in t if c.isupper()) > len(t) * 0.7,
    "multi_issue": lambda t: sum(
        1 for intent, kws in INTENT_KEYWORDS.items()
        if any(kw in t.lower() for kw in kws)
    ) >= 2,
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load twcs.csv and filter to AppleSupport conversations."""
    print(f"[SAMPLE] Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[SAMPLE] Loaded {len(df)} rows")

    # Get AppleSupport response tweet IDs
    apple_responses = df[df["author_id"] == "AppleSupport"]
    apple_response_ids = set(apple_responses["tweet_id"].values)
    print(f"[SAMPLE] AppleSupport responses: {len(apple_response_ids)}")

    # Get customer messages responding to AppleSupport
    customer_msgs = df[
        (df["in_response_to_tweet_id"].isin(apple_response_ids))
        & (df["inbound"] == True)
    ].copy()
    print(f"[SAMPLE] Customer messages to AppleSupport: {len(customer_msgs)}")

    return df, customer_msgs, apple_response_ids


def classify_intent_keyword(text: str) -> str:
    """Pre-classify intent using keyword matching (fast, no LLM)."""
    text_lower = text.lower()
    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unclear"

    return max(scores, key=scores.get)


def detect_edge_cases(text: str) -> list:
    """Detect edge case types in a message."""
    detected = []
    for case_name, detector in EDGE_CASES.items():
        try:
            if detector(text):
                detected.append(case_name)
        except Exception:
            pass
    return detected


def get_thread_context(df: pd.DataFrame, tweet_id: int, max_context: int = 3) -> list:
    """Reconstruct thread context by following in_response_to chains."""
    context = []
    current_id = tweet_id

    for _ in range(max_context):
        # Find the tweet that this one is responding to
        parent_rows = df[df["tweet_id"] == current_id]
        if parent_rows.empty:
            break

        parent_id = parent_rows.iloc[0].get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "":
            break

        try:
            parent_id = int(float(parent_id))
        except (ValueError, TypeError):
            break

        parent_rows = df[df["tweet_id"] == parent_id]
        if parent_rows.empty:
            break

        parent = parent_rows.iloc[0]
        context.insert(0, {
            "author": parent["author_id"],
            "text": str(parent["text"]),
            "tweet_id": parent_id,
        })
        current_id = parent_id

    return context


def get_brand_response(df: pd.DataFrame, tweet_id: int) -> str | None:
    """Get AppleSupport's response to a customer message, if any."""
    responses = df[
        (df["in_response_to_tweet_id"] == tweet_id)
        & (df["author_id"] == "AppleSupport")
    ]
    if len(responses) > 0:
        return str(responses.iloc[0]["text"])
    return None


def sample_eval_set(
    df: pd.DataFrame,
    customer_msgs: pd.DataFrame,
    apple_response_ids: set,
    num_samples: int = 200,
    seed: int = 42,
) -> list:
    """Sample eval candidates stratified across intents + edge cases."""
    np.random.seed(seed)

    # Pre-bucket messages by intent
    print("[SAMPLE] Pre-bucketing messages by intent...")
    customer_msgs = customer_msgs.copy()
    customer_msgs["detected_intent"] = customer_msgs["text"].apply(
        lambda t: classify_intent_keyword(str(t))
    )
    customer_msgs["edge_cases"] = customer_msgs["text"].apply(
        lambda t: detect_edge_cases(str(t))
    )

    # Count per intent
    intent_counts = customer_msgs["detected_intent"].value_counts()
    print(f"[SAMPLE] Intent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

    # Calculate samples per intent
    intents = [i for i in INTENT_KEYWORDS.keys() if i in intent_counts.index]
    per_intent = num_samples // (len(intents) + 1)  # +1 for unclear
    edge_case_budget = min(50, num_samples // 4)

    sampled = []
    sampled_ids = set()

    # Sample per intent
    for intent in intents:
        intent_msgs = customer_msgs[customer_msgs["detected_intent"] == intent]
        n = min(per_intent, len(intent_msgs))
        if n > 0:
            samples = intent_msgs.sample(n=n, random_state=seed)
            for _, row in samples.iterrows():
                if row["tweet_id"] not in sampled_ids:
                    sampled.append(row)
                    sampled_ids.add(row["tweet_id"])

    # Sample unclear
    unclear_msgs = customer_msgs[customer_msgs["detected_intent"] == "unclear"]
    n = min(per_intent, len(unclear_msgs))
    if n > 0:
        samples = unclear_msgs.sample(n=n, random_state=seed)
        for _, row in samples.iterrows():
            if row["tweet_id"] not in sampled_ids:
                sampled.append(row)
                sampled_ids.add(row["tweet_id"])

    # Oversample edge cases
    print(f"[SAMPLE] Oversampling {edge_case_budget} edge cases...")
    edge_case_pool = customer_msgs[
        customer_msgs["edge_cases"].apply(len) > 0
    ]
    edge_case_pool = edge_case_pool[~edge_case_pool["tweet_id"].isin(sampled_ids)]

    if len(edge_case_pool) > 0:
        n = min(edge_case_budget, len(edge_case_pool))
        edge_samples = edge_case_pool.sample(n=n, random_state=seed)
        for _, row in edge_samples.iterrows():
            if row["tweet_id"] not in sampled_ids:
                sampled.append(row)
                sampled_ids.add(row["tweet_id"])

    # Pad with random samples if needed
    remaining = num_samples - len(sampled)
    if remaining > 0:
        unsampled = customer_msgs[~customer_msgs["tweet_id"].isin(sampled_ids)]
        if len(unsampled) >= remaining:
            padding = unsampled.sample(n=remaining, random_state=seed)
            for _, row in padding.iterrows():
                sampled.append(row)
                sampled_ids.add(row["tweet_id"])

    print(f"[SAMPLE] Total sampled: {len(sampled)}")
    return sampled


def build_eval_candidates(df: pd.DataFrame, sampled: list, output_path: str):
    """Build eval_candidates.jsonl with thread context."""
    print(f"[SAMPLE] Building {output_path}...")
    candidates = []

    for i, row in enumerate(sampled):
        tweet_id = row["tweet_id"]
        text = str(row["text"])

        # Get thread context
        context = get_thread_context(df, tweet_id, max_context=3)

        # Get brand response
        brand_response = get_brand_response(df, tweet_id)

        # Detect edge cases
        edge_cases = detect_edge_cases(text)

        candidate = {
            "tweet_id": int(tweet_id),
            "text": text,
            "thread_context": context,
            "brand_response": brand_response,
            "detected_intent": classify_intent_keyword(text),
            "edge_cases": edge_cases,
            "sample_index": i,
        }
        candidates.append(candidate)

    # Write to JSONL
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"[SAMPLE] Wrote {len(candidates)} candidates to {output_path}")

    # Print summary
    intents = {}
    edge_count = 0
    for c in candidates:
        intent = c["detected_intent"]
        intents[intent] = intents.get(intent, 0) + 1
        if c["edge_cases"]:
            edge_count += 1

    print(f"\n[SAMPLE] Summary:")
    print(f"  Total: {len(candidates)}")
    print(f"  Edge cases: {edge_count}")
    print(f"  Intent distribution:")
    for intent, count in sorted(intents.items()):
        print(f"    {intent}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Sample eval candidates for golden set labeling")
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).parent.parent / "data" / "twcs.csv"),
        help="Path to twcs.csv",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "eval_candidates.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df, customer_msgs, apple_response_ids = load_dataset(args.csv_path)
    sampled = sample_eval_set(df, customer_msgs, apple_response_ids, args.num_samples, args.seed)
    build_eval_candidates(df, sampled, args.output)


if __name__ == "__main__":
    main()
