"""Split golden evaluation set into training and test sets.

Creates a stratified split of the 215 human-labeled messages:
- Training set: 170 messages (79%) - used to build FAISS k-NN database
- Test set: 45 messages (21%) - used to evaluate hybrid classifier

The split preserves intent distribution to ensure fair evaluation.

Usage:
    python split_dataset.py                    # Default 79/21 split
    python split_dataset.py --test-size 0.25   # 75/25 split
    python split_dataset.py --seed 123         # Custom random seed
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict


def load_golden_set(path: str) -> list:
    """Load human-labeled messages from golden eval set."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("labeled_by") == "human":
                    data.append(entry)
    return data


def stratified_split(data: list, test_size: float = 0.21, seed: int = 42) -> tuple:
    """Split data into train/test with stratification by intent.
    
    Args:
        data: List of labeled messages
        test_size: Fraction of data for test set (default 0.21 = 45/215)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(seed)
    
    # Group by intent
    by_intent = defaultdict(list)
    for item in data:
        intent = item.get("labeled_intent", "unknown")
        by_intent[intent].append(item)
    
    train = []
    test = []
    
    for intent, items in by_intent.items():
        # Shuffle items for this intent
        random.shuffle(items)
        
        # Calculate split point
        n_test = max(1, round(len(items) * test_size))
        n_train = len(items) - n_test
        
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    
    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test


def save_jsonl(data: list, path: str):
    """Save data to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def print_statistics(train: list, test: list):
    """Print split statistics."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)
    
    # Overall counts
    print(f"\nTotal messages: {len(train) + len(test)}")
    print(f"Training set: {len(train)} ({len(train)/(len(train)+len(test))*100:.1f}%)")
    print(f"Test set: {len(test)} ({len(test)/(len(train)+len(test))*100:.1f}%)")
    
    # Intent distribution
    train_intents = Counter(item["labeled_intent"] for item in train)
    test_intents = Counter(item["labeled_intent"] for item in test)
    all_intents = sorted(set(list(train_intents.keys()) + list(test_intents.keys())))
    
    print("\n" + "-" * 60)
    print("INTENT DISTRIBUTION")
    print("-" * 60)
    print(f"{'Intent':<30} {'Train':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    for intent in all_intents:
        t_count = train_intents.get(intent, 0)
        te_count = test_intents.get(intent, 0)
        total = t_count + te_count
        print(f"{intent:<30} {t_count:>8} {te_count:>8} {total:>8}")
    
    print("-" * 60)
    print(f"{'TOTAL':<30} {len(train):>8} {len(test):>8} {len(train)+len(test):>8}")
    
    # Escalation distribution
    train_escalate = sum(1 for item in train if item.get("labeled_escalate"))
    test_escalate = sum(1 for item in test if item.get("labeled_escalate"))
    
    print("\n" + "-" * 60)
    print("ESCALATION DISTRIBUTION")
    print("-" * 60)
    print(f"{'Set':<30} {'Escalate':>10} {'No Escalate':>12} {'Total':>8}")
    print("-" * 60)
    print(f"{'Train':<30} {train_escalate:>10} {len(train)-train_escalate:>12} {len(train):>8}")
    print(f"{'Test':<30} {test_escalate:>10} {len(test)-test_escalate:>12} {len(test):>8}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Split golden eval set into train/test")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent.parent / "data" / "golden_eval_set.jsonl"),
        help="Path to golden evaluation set"
    )
    parser.add_argument(
        "--train-output",
        default=str(Path(__file__).parent.parent / "data" / "train_set.jsonl"),
        help="Output path for training set"
    )
    parser.add_argument(
        "--test-output",
        default=str(Path(__file__).parent.parent / "data" / "test_set.jsonl"),
        help="Output path for test set"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.21,
        help="Fraction of data for test set (default: 0.21 = 45/215)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Load data
    print(f"Loading golden set from {args.input}...")
    data = load_golden_set(args.input)
    print(f"Loaded {len(data)} human-labeled messages")
    
    # Split
    print(f"\nCreating stratified split (test_size={args.test_size}, seed={args.seed})...")
    train, test = stratified_split(data, test_size=args.test_size, seed=args.seed)
    
    # Save
    print(f"\nSaving training set to {args.train_output}...")
    save_jsonl(train, args.train_output)
    
    print(f"Saving test set to {args.test_output}...")
    save_jsonl(test, args.test_output)
    
    # Print statistics
    print_statistics(train, test)
    
    print("\n✅ Split complete!")
    print(f"   Training set: {args.train_output}")
    print(f"   Test set: {args.test_output}")


if __name__ == "__main__":
    main()
