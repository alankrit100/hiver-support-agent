"""Evaluation harness for the Apple Support Agent pipeline.

Computes classification metrics, escalation metrics, and compares against baselines.
Generates a comprehensive evaluation report.

Usage:
    python evaluate.py                    # Run all evaluations
    python evaluate.py --mode classification  # Classification only
    python evaluate.py --mode escalation      # Escalation only
    python evaluate.py --mode baselines       # Baselines only
    python evaluate.py --mode reply_quality   # LLM-as-judge only
    python evaluate.py --mode hybrid          # Hybrid classifier evaluation
    python evaluate.py --mode compare         # Compare LLM-only vs Hybrid
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def load_golden_set(path: str) -> List[dict]:
    """Load the golden evaluation set."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Only include entries with human labels
                if entry.get("labeled_by") == "human":
                    data.append(entry)
    return data


def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], intents: List[str]
) -> Dict:
    """Compute precision, recall, F1 per intent and confusion matrix."""
    # Initialize metrics
    metrics = {}
    confusion = defaultdict(lambda: defaultdict(int))

    # Count true positives, false positives, false negatives per intent
    for intent in intents:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != intent and p == intent)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p != intent)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == intent)

        metrics[intent] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    # Build confusion matrix
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0

    return {
        "per_intent": metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "accuracy": round(accuracy, 3),
        "total": len(y_true),
        "correct": correct,
    }


def compute_escalation_metrics(
    y_true_escalate: List[bool], y_pred_escalate: List[bool]
) -> Dict:
    """Compute escalation precision, recall, and false negatives."""
    tp = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and p)
    fp = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if not t and p)
    fn = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p)
    tn = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true_escalate) if y_true_escalate else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,  # WORSE than false positives
        "true_negatives": tn,
        "total": len(y_true_escalate),
    }


def display_classification_results(results: Dict, intents: List[str]):
    """Display classification metrics in a formatted table."""
    console.print(Panel("[bold green]Classification Metrics[/]", border_style="green"))

    # Per-intent metrics table
    table = Table(title="Per-Intent Metrics")
    table.add_column("Intent", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Support", justify="right")

    for intent in intents:
        m = results["per_intent"][intent]
        f1_color = "green" if m["f1"] >= 0.8 else "yellow" if m["f1"] >= 0.6 else "red"
        table.add_row(
            intent,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"[{f1_color}]{m['f1']:.3f}[/{f1_color}]",
            str(m["support"]),
        )

    # Overall accuracy row
    table.add_section()
    acc_color = "green" if results["accuracy"] >= 0.8 else "yellow" if results["accuracy"] >= 0.6 else "red"
    table.add_row(
        "[bold]OVERALL[/]",
        "",
        "",
        f"[{acc_color}]{results['accuracy']:.3f}[/{acc_color}]",
        str(results["total"]),
    )

    console.print(table)

    # Confusion matrix
    console.print("\n[bold yellow]Confusion Matrix:[/]")
    conf_table = Table(show_header=True, show_lines=True)
    conf_table.add_column("True \\ Pred", style="cyan")

    for intent in intents:
        conf_table.add_column(intent[:12], justify="right")

    for true_intent in intents:
        row = [true_intent]
        for pred_intent in intents:
            count = results["confusion_matrix"].get(true_intent, {}).get(pred_intent, 0)
            if true_intent == pred_intent:
                row.append(f"[green]{count}[/green]")
            elif count > 0:
                row.append(f"[red]{count}[/red]")
            else:
                row.append("0")
        conf_table.add_row(*row)

    console.print(conf_table)


def display_escalation_results(results: Dict):
    """Display escalation metrics."""
    console.print(Panel("[bold red]Escalation Metrics[/]", border_style="red"))

    table = Table(title="Escalation Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Precision", f"{results['precision']:.3f}")
    table.add_row("Recall", f"{results['recall']:.3f}")
    table.add_row("F1-Score", f"{results['f1']:.3f}")
    table.add_row("Accuracy", f"{results['accuracy']:.3f}")
    table.add_section()
    table.add_row("True Positives (correctly escalated)", str(results["true_positives"]))
    table.add_row("False Positives (unnecessary escalation)", str(results["false_positives"]))
    table.add_row(
        "[bold red]False Negatives (MISSED escalations)[/]",
        str(results["false_negatives"]),
    )
    table.add_row("True Negatives (correctly not escalated)", str(results["true_negatives"]))
    table.add_section()
    table.add_row("Total Messages", str(results["total"]))

    console.print(table)

    # Warning about false negatives
    if results["false_negatives"] > 0:
        console.print(
            f"\n[bold red]⚠ WARNING: {results['false_negatives']} escalations were MISSED. "
            f"Missed escalations are worse than unnecessary ones — these need investigation.[/]"
        )


def load_progress(progress_path: str) -> dict:
    """Load progress from file if it exists."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"last_completed_index": -1, "results": []}


def save_progress(progress_path: str, progress: dict):
    """Save progress to file."""
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def run_classification_evaluation(golden_set: List[dict], intents: List[str]) -> Dict:
    """Run full classification evaluation using the pipeline."""
    from pipeline import SupportPipeline

    console.print("\n[bold blue]Running Classification Evaluation...[/]")

    # Initialize pipeline
    pipeline = SupportPipeline()

    # Progress file path
    progress_path = str(Path(__file__).parent.parent / "data" / "eval_progress.json")
    progress = load_progress(progress_path)
    start_index = progress["last_completed_index"] + 1

    if start_index > 0:
        console.print(f"[yellow]Resuming from message {start_index + 1}...[/]")
        # Load existing results
        y_true = [r["true_intent"] for r in progress["results"]]
        y_pred = [r["pred_intent"] for r in progress["results"]]
        y_true_escalate = [r["true_escalate"] for r in progress["results"]]
        y_pred_escalate = [r["pred_escalate"] for r in progress["results"]]
        results_log = [r for r in progress["results"] if r["true_intent"] != r["pred_intent"]]
    else:
        y_true = []
        y_pred = []
        y_true_escalate = []
        y_pred_escalate = []
        results_log = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Processing messages...", total=len(golden_set))

        for idx, entry in enumerate(golden_set):
            # Skip already processed messages
            if idx < start_index:
                progress_bar.update(task, advance=1)
                continue

            message = entry["text"]
            true_intent = entry["labeled_intent"]
            true_escalate = entry["labeled_escalate"]

            # Run pipeline with error handling
            try:
                result = pipeline.process_message(message)
                pred_intent = result["intent"]
                pred_escalate = result["escalate"]
            except Exception as e:
                print(f"\n[ERROR] Message {idx + 1} failed: {e}")
                print(f"[ERROR] Stopping evaluation. Resume from message {idx + 1}.")
                # Save progress and exit
                save_progress(progress_path, {
                    "last_completed_index": idx - 1,
                    "results": progress["results"]
                })
                raise

            y_true.append(true_intent)
            y_pred.append(pred_intent)
            y_true_escalate.append(true_escalate)
            y_pred_escalate.append(pred_escalate)

            # Log mismatches
            result_entry = {
                "tweet_id": entry["tweet_id"],
                "text": message[:100],
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "true_escalate": true_escalate,
                "pred_escalate": pred_escalate,
            }
            if true_intent != pred_intent:
                results_log.append(result_entry)

            # Save progress every 2 messages
            progress["results"].append(result_entry)
            if (idx + 1) % 2 == 0:
                progress["last_completed_index"] = idx
                save_progress(progress_path, progress)

            progress_bar.update(task, advance=1)

    # Save final progress
    progress["last_completed_index"] = len(golden_set) - 1
    save_progress(progress_path, progress)

    # Compute metrics
    classification_metrics = compute_classification_metrics(y_true, y_pred, intents)
    escalation_metrics = compute_escalation_metrics(y_true_escalate, y_pred_escalate)

    # Add mismatch details
    classification_metrics["mismatches"] = results_log[:50]  # Top 50 mismatches

    return {
        "classification": classification_metrics,
        "escalation": escalation_metrics,
    }


def run_hybrid_evaluation(golden_set: List[dict], intents: List[str]) -> Dict:
    """Run evaluation using the hybrid classifier.
    
    This evaluates on the test set (45 messages) using the hybrid approach.
    The hybrid classifier uses semantic search + LLM for classification.
    """
    from hybrid_classifier import HybridClassifier

    console.print("\n[bold blue]Running Hybrid Classifier Evaluation...[/]")
    console.print("[bold cyan]Using test set (45 messages) with BGE-M3 + LLM[/]")

    # Initialize hybrid classifier
    classifier = HybridClassifier()

    y_true = []
    y_pred = []
    y_true_escalate = []
    y_pred_escalate = []
    results_log = []
    method_counts = defaultdict(int)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Processing messages with hybrid classifier...", total=len(golden_set))

        for idx, entry in enumerate(golden_set):
            message = entry["text"]
            true_intent = entry["labeled_intent"]
            true_escalate = entry["labeled_escalate"]

            # Run hybrid classifier with error handling
            try:
                result = classifier.classify(message)
                pred_intent = result["intent"]
                method = result.get("method", "unknown")
                method_counts[method] += 1
            except Exception as e:
                print(f"\n[ERROR] Message {idx + 1} failed: {e}")
                pred_intent = "unclear"
                method = "error"

            y_true.append(true_intent)
            y_pred.append(pred_intent)
            y_true_escalate.append(true_escalate)
            # Hybrid classifier doesn't predict escalation, use rule-based
            y_pred_escalate.append(true_escalate)  # Placeholder

            # Log mismatches
            result_entry = {
                "tweet_id": entry["tweet_id"],
                "text": message[:100],
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "method": method,
                "true_escalate": true_escalate,
                "pred_escalate": true_escalate,
            }
            if true_intent != pred_intent:
                results_log.append(result_entry)

            progress_bar.update(task, advance=1)

    # Compute metrics
    classification_metrics = compute_classification_metrics(y_true, y_pred, intents)
    
    # Add mismatch details
    classification_metrics["mismatches"] = results_log[:50]
    classification_metrics["method_distribution"] = dict(method_counts)

    return {
        "classification": classification_metrics,
    }


def run_comparison_evaluation(
    llm_results: Dict, 
    hybrid_results: Dict, 
    intents: List[str]
) -> Dict:
    """Compare LLM-only vs Hybrid classifier results."""
    console.print(Panel("[bold cyan]COMPARISON: LLM-Only vs Hybrid[/]", border_style="cyan"))

    # Extract metrics
    llm_acc = llm_results["classification"]["accuracy"]
    hybrid_acc = hybrid_results["classification"]["accuracy"]
    improvement = hybrid_acc - llm_acc

    # Display comparison table
    table = Table(title="Accuracy Comparison")
    table.add_column("Approach", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    table.add_row(
        "LLM-Only",
        f"{llm_acc:.3f}",
        str(llm_results["classification"]["correct"]),
        str(llm_results["classification"]["total"]),
    )
    table.add_row(
        "Hybrid",
        f"{hybrid_acc:.3f}",
        str(hybrid_results["classification"]["correct"]),
        str(hybrid_results["classification"]["total"]),
    )
    table.add_section()
    improvement_color = "green" if improvement > 0 else "red"
    table.add_row(
        f"[{improvement_color}]Improvement[/{improvement_color}]",
        f"[{improvement_color}]{improvement:+.3f}[/{improvement_color}]",
        "",
        "",
    )

    console.print(table)

    # Per-intent comparison
    console.print("\n[bold]Per-Intent Comparison:[/]")
    intent_table = Table(title="F1-Score Comparison")
    intent_table.add_column("Intent", style="cyan")
    intent_table.add_column("LLM-Only F1", justify="right")
    intent_table.add_column("Hybrid F1", justify="right")
    intent_table.add_column("Change", justify="right")

    for intent in intents:
        llm_f1 = llm_results["classification"]["per_intent"][intent]["f1"]
        hybrid_f1 = hybrid_results["classification"]["per_intent"][intent]["f1"]
        change = hybrid_f1 - llm_f1
        change_color = "green" if change > 0 else "red" if change < 0 else "yellow"
        intent_table.add_row(
            intent,
            f"{llm_f1:.3f}",
            f"{hybrid_f1:.3f}",
            f"[{change_color}]{change:+.3f}[/{change_color}]",
        )

    console.print(intent_table)

    return {
        "llm_accuracy": llm_acc,
        "hybrid_accuracy": hybrid_acc,
        "improvement": improvement,
    }


def display_mismatch_analysis(results: Dict):
    """Display analysis of classification mismatches."""
    mismatches = results["classification"].get("mismatches", [])

    if not mismatches:
        console.print("\n[green]No mismatches found![/]")
        return

    console.print(Panel("[bold yellow]Top Classification Mismatches[/]", border_style="yellow"))

    # Count mismatches by pair
    pair_counts = defaultdict(int)
    for m in mismatches:
        pair = (m["true_intent"], m["pred_intent"])
        pair_counts[pair] += 1

    # Sort by count
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    table = Table(title="Most Common Confusions")
    table.add_column("True Intent", style="cyan")
    table.add_column("Predicted As", style="red")
    table.add_column("Count", justify="right")

    for (true, pred), count in sorted_pairs[:10]:
        table.add_row(true, pred, str(count))

    console.print(table)

    # Show example mismatches
    console.print("\n[bold]Example Mismatches:[/]")
    for m in mismatches[:5]:
        console.print(f"  [cyan]True:[/] {m['true_intent']}")
        console.print(f"  [red]Predicted:[/] {m['pred_intent']}")
        console.print(f"  [dim]Message:[/] {m['text'][:80]}...")
        console.print()


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Apple Support Agent Pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "classification", "escalation", "baselines", "reply_quality", "hybrid", "compare"],
        default="all",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--golden-set",
        default=str(Path(__file__).parent.parent / "data" / "golden_eval_set.jsonl"),
        help="Path to golden evaluation set",
    )
    parser.add_argument(
        "--test-set",
        default=str(Path(__file__).parent.parent / "data" / "test_set.jsonl"),
        help="Path to test set for hybrid evaluation",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "eval_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for evaluation (e.g., 50 for quick test)",
    )
    args = parser.parse_args()

    # Load intents
    intents_path = str(Path(__file__).parent / "intent_definitions.json")
    with open(intents_path) as f:
        intent_data = json.load(f)
    intents = [i["name"] for i in intent_data["intents"]]

    # Load golden set
    console.print("[bold blue]Loading golden evaluation set...[/]")
    golden_set = load_golden_set(args.golden_set)
    console.print(f"[green]{len(golden_set)} labeled messages loaded[/]\n")

    if not golden_set:
        console.print("[red]No human-labeled messages found in golden set![/]")
        sys.exit(1)

    # Sample if requested
    if args.sample and args.sample < len(golden_set):
        import random
        golden_set = random.sample(golden_set, args.sample)
        console.print(f"[yellow]Sampled {args.sample} messages for evaluation[/]\n")

    # Run evaluation based on mode
    results = {}

    if args.mode in ("all", "classification", "escalation"):
        eval_results = run_classification_evaluation(golden_set, intents)

        if args.mode in ("all", "classification"):
            results["classification"] = eval_results["classification"]
            display_classification_results(eval_results["classification"], intents)
            display_mismatch_analysis(eval_results)

        if args.mode in ("all", "escalation"):
            results["escalation"] = eval_results["escalation"]
            display_escalation_results(eval_results["escalation"])

    # Run baselines
    if args.mode in ("all", "baselines"):
        from baselines import run_baselines

        console.print("\n[bold blue]Running Baselines...[/]")
        baseline_results = run_baselines(golden_set, intents)
        results["baselines"] = baseline_results

    # Run LLM-as-judge
    if args.mode in ("all", "reply_quality"):
        from llm_judge import run_llm_judge

        console.print("\n[bold blue]Running LLM-as-Judge...[/]")
        judge_results = run_llm_judge(golden_set)
        results["reply_quality"] = judge_results

    # Run hybrid evaluation
    if args.mode in ("hybrid", "compare"):
        # Load test set for hybrid evaluation
        console.print("[bold blue]Loading test set for hybrid evaluation...[/]")
        test_set = load_golden_set(args.test_set)
        console.print(f"[green]{len(test_set)} test messages loaded[/]\n")
        
        hybrid_results = run_hybrid_evaluation(test_set, intents)
        results["hybrid"] = hybrid_results["classification"]
        display_classification_results(hybrid_results["classification"], intents)
        display_mismatch_analysis(hybrid_results)

    # Compare LLM-only vs Hybrid
    if args.mode == "compare":
        # We need LLM-only results on the same test set
        console.print("\n[bold blue]Running LLM-Only on test set for comparison...[/]")
        llm_test_results = run_classification_evaluation(test_set, intents)
        results["llm_only_test"] = llm_test_results["classification"]
        
        # Run comparison
        comparison = run_comparison_evaluation(
            llm_test_results, 
            hybrid_results, 
            intents
        )
        results["comparison"] = comparison

    # Save results
    save_results(results, args.output)

    # Print summary
    console.print(Panel("[bold green]Evaluation Complete![/]", border_style="green"))
    console.print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
