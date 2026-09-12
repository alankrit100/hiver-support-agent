"""Baseline models for comparison.

Implements trivial and simple baselines to compare against the full pipeline.
Shows if the pipeline adds value over simpler approaches.

Baselines:
1. Trivial: Canned reply for everything + majority-class intent guess
2. Simple: TF-IDF + logistic regression classifier + template replies
"""

import json
from collections import Counter
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class TrivialBaseline:
    """Baseline that uses same reply for everything."""

    CANNED_REPLY = (
        "Thank you for contacting Apple Support. "
        "We're sorry to hear you're experiencing issues. "
        "Please try restarting your device and see if that helps. "
        "If the problem persists, please contact us again."
    )

    def __init__(self, intents: List[str]):
        self.intents = intents
        self.majority_intent = None

    def fit(self, golden_set: List[dict]):
        """Learn majority class from training data."""
        intent_counts = Counter(entry["labeled_intent"] for entry in golden_set)
        self.majority_intent = intent_counts.most_common(1)[0][0]
        console.print(f"  Trivial baseline: majority class = {self.majority_intent}")

    def predict(self, message: str) -> Tuple[str, str, bool]:
        """Always return majority class and canned reply."""
        return self.majority_intent, self.CANNED_REPLY, False

    def evaluate(self, golden_set: List[dict]) -> Dict:
        """Evaluate trivial baseline on golden set."""
        y_true = [entry["labeled_intent"] for entry in golden_set]
        y_pred = [self.predict(entry["text"])[0] for entry in golden_set]

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0

        # Per-intent metrics
        metrics = {}
        for intent in self.intents:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != intent and p == intent)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p != intent)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[intent] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }

        return {
            "name": "Trivial (canned reply)",
            "accuracy": round(accuracy, 3),
            "per_intent": metrics,
        }


class SimpleBaseline:
    """TF-IDF + logistic regression classifier with template replies."""

    # Intent-specific templates
    TEMPLATES = {
        "ios_update_issues": (
            "We understand you're having issues with iOS updates. "
            "Please try restarting your device and checking your internet connection. "
            "You can also try updating via iTunes on a computer."
        ),
        "battery_drain_after_update": (
            "We understand you're experiencing battery drain. "
            "Please check Battery Usage in Settings to see which apps are consuming power. "
            "Try turning off Background App Refresh for unused apps."
        ),
        "device_crash_freeze": (
            "We understand your device is crashing or freezing. "
            "Please try force restarting your device by holding Power + Home for 10 seconds. "
            "If the issue persists, please contact us for further assistance."
        ),
        "app_malfunction": (
            "We understand you're having issues with an app. "
            "Please try force quitting the app and reopening it. "
            "You can also try deleting and reinstalling the app."
        ),
        "account_access": (
            "We understand you're having trouble with your account. "
            "Please try signing out and signing back in. "
            "You can reset your password at iforgot.apple.com."
        ),
        "unclear": (
            "Thank you for contacting Apple Support. "
            "Could you please provide more details about the issue you're experiencing? "
            "This will help us assist you better."
        ),
    }

    def __init__(self, intents: List[str]):
        self.intents = intents
        self.vectorizer = None
        self.classifier = None

    def fit(self, golden_set: List[dict]):
        """Train TF-IDF + logistic regression on golden set."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = [entry["text"] for entry in golden_set]
        labels = [entry["labeled_intent"] for entry in golden_set]

        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)

        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
        self.classifier.fit(X, labels)

        console.print(f"  Simple baseline: trained on {len(texts)} examples")

    def predict(self, message: str) -> Tuple[str, str, bool]:
        """Predict intent using TF-IDF + logistic regression."""
        if not self.vectorizer or not self.classifier:
            return "unclear", self.TEMPLATES["unclear"], False

        X = self.vectorizer.transform([message])
        intent = self.classifier.predict(X)[0]
        reply = self.TEMPLATES.get(intent, self.TEMPLATES["unclear"])

        # Simple escalation heuristic (no LLM)
        escalate = False
        message_lower = message.lower()
        escalation_keywords = ["angry", "furious", "frustrated", "unacceptable", "sue", "lawyer"]
        if any(kw in message_lower for kw in escalation_keywords):
            escalate = True

        return intent, reply, escalate

    def evaluate(self, golden_set: List[dict]) -> Dict:
        """Evaluate simple baseline on golden set."""
        y_true = [entry["labeled_intent"] for entry in golden_set]
        y_pred = [self.predict(entry["text"])[0] for entry in golden_set]

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0

        # Per-intent metrics
        metrics = {}
        for intent in self.intents:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != intent and p == intent)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p != intent)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[intent] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }

        # Escalation metrics
        y_true_escalate = [entry["labeled_escalate"] for entry in golden_set]
        y_pred_escalate = [self.predict(entry["text"])[2] for entry in golden_set]

        tp_e = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and p)
        fp_e = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if not t and p)
        fn_e = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p)

        precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0
        recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0

        return {
            "name": "Simple (TF-IDF + logistic regression)",
            "accuracy": round(accuracy, 3),
            "per_intent": metrics,
            "escalation": {
                "precision": round(precision_e, 3),
                "recall": round(recall_e, 3),
            },
        }


def display_baseline_results(results: List[Dict]):
    """Display baseline comparison results."""
    console.print(Panel("[bold blue]Baseline Comparison[/]", border_style="blue"))

    # Main comparison table
    table = Table(title="Method Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Escalation Precision", justify="right")
    table.add_column("Escalation Recall", justify="right")

    for result in results:
        # Calculate macro F1
        f1_scores = [m["f1"] for m in result["per_intent"].values()]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        # Get escalation metrics if available
        esc_precision = result.get("escalation", {}).get("precision", "N/A")
        esc_recall = result.get("escalation", {}).get("recall", "N/A")

        # Color code accuracy
        acc = result["accuracy"]
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"

        table.add_row(
            result["name"],
            f"[{acc_color}]{acc:.3f}[/{acc_color}]",
            f"{macro_f1:.3f}",
            f"{esc_precision:.3f}" if isinstance(esc_precision, float) else esc_precision,
            f"{esc_recall:.3f}" if isinstance(esc_recall, float) else esc_recall,
        )

    console.print(table)

    # Highlight differences
    if len(results) >= 2:
        pipeline_result = next((r for r in results if "Pipeline" in r["name"]), None)
        trivial_result = next((r for r in results if "Trivial" in r["name"]), None)
        simple_result = next((r for r in results if "Simple" in r["name"]), None)

        if pipeline_result and trivial_result:
            diff = pipeline_result["accuracy"] - trivial_result["accuracy"]
            console.print(f"\n[green]Pipeline beats Trivial baseline by {diff:.1%}[/]")

        if pipeline_result and simple_result:
            diff = pipeline_result["accuracy"] - simple_result["accuracy"]
            console.print(f"[green]Pipeline beats Simple baseline by {diff:.1%}[/]")


def run_baselines(golden_set: List[dict], intents: List[str]) -> Dict:
    """Run all baselines and return results."""
    # Trivial baseline
    console.print("  Running trivial baseline...")
    trivial = TrivialBaseline(intents)
    trivial.fit(golden_set)
    trivial_result = trivial.evaluate(golden_set)

    # Simple baseline
    console.print("  Running simple baseline...")
    simple = SimpleBaseline(intents)
    simple.fit(golden_set)
    simple_result = simple.evaluate(golden_set)

    # Display results
    display_baseline_results([trivial_result, simple_result])

    return {
        "trivial": trivial_result,
        "simple": simple_result,
    }


if __name__ == "__main__":
    # Test baselines
    from pathlib import Path

    golden_set_path = str(Path(__file__).parent.parent / "data" / "golden_eval_set.jsonl")
    intents_path = str(Path(__file__).parent / "intent_definitions.json")

    # Load data
    with open(intents_path) as f:
        intent_data = json.load(f)
    intents = [i["name"] for i in intent_data["intents"]]

    golden_set = []
    with open(golden_set_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("labeled_by") == "human":
                    golden_set.append(entry)

    console.print(f"Loaded {len(golden_set)} labeled messages\n")
    run_baselines(golden_set, intents)
