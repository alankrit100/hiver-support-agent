"""Human scoring interface for reply quality.

Interactive script to manually score 28 replies on 5 dimensions.

Usage:
    python human_scorer.py              # Start scoring
    python human_scorer.py --resume     # Resume from last saved position
    python human_scorer.py --skip 10    # Skip first 10
"""

import json
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

console = Console()

DIMENSIONS = [
    ("groundedness", "Is it based on real examples?"),
    ("relevance", "Does it address the actual issue?"),
    ("correctness", "Is the advice technically accurate?"),
    ("tone", "Professional and empathetic?"),
    ("actionability", "Clear next steps?"),
]

RUBRIC = """
## Scoring Rubric

| Score | Meaning |
|-------|---------|
| 1 | Poor - Does not meet the criteria |
| 2 | Below Average - Partially meets criteria |
| 3 | Average - Adequately meets criteria |
| 4 | Good - Strongly meets criteria |
| 5 | Excellent - Fully meets criteria |
"""


def load_template(path: str) -> list:
    """Load the scoring template."""
    with open(path) as f:
        return json.load(f)


def load_progress(progress_path: str) -> dict:
    """Load progress from file if it exists."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"last_completed_index": -1, "scores": []}


def save_progress(progress_path: str, progress: dict):
    """Save progress to file."""
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def display_rubric():
    """Display the scoring rubric."""
    console.print(Panel(Markdown(RUBRIC), title="Scoring Guide", border_style="blue"))


def display_message(entry: dict, index: int, total: int):
    """Display a message for scoring."""
    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]Message {index + 1} of {total}[/]")
    console.print(f"{'='*60}\n")
    
    console.print(f"[bold]Tweet ID:[/] {entry['tweet_id']}")
    console.print(f"[bold]Intent:[/] {entry['intent']}")
    console.print()
    
    console.print("[bold yellow]Customer Message:[/]")
    console.print(f"  {entry['message']}")
    console.print()
    
    console.print("[bold green]Brand Reply:[/]")
    console.print(f"  {entry['reply']}")
    console.print()


def score_message() -> dict:
    """Get scores for a single message."""
    scores = {}
    
    for dim, description in DIMENSIONS:
        console.print(f"[bold cyan]{dim}[/] - {description}")
        console.print("  Score (1-5): ", end="")
        
        while True:
            try:
                score = int(input())
                if 1 <= score <= 5:
                    scores[dim] = score
                    break
                else:
                    console.print("  [red]Please enter 1-5:[/] ", end="")
            except ValueError:
                console.print("  [red]Please enter a number:[/] ", end="")
    
    return scores


def compute_agreement(llm_scores: list, human_scores: list) -> dict:
    """Compute agreement metrics between LLM and human scores."""
    import numpy as np
    
    dims = ["groundedness", "relevance", "correctness", "tone", "actionability"]
    
    agreement = {}
    for dim in dims:
        llm_vals = [s[dim] for s in llm_scores if dim in s]
        human_vals = [s[dim] for s in human_scores if dim in s]
        
        if len(llm_vals) != len(human_vals):
            continue
        
        llm_arr = np.array(llm_vals, dtype=float)
        human_arr = np.array(human_vals, dtype=float)
        
        # Absolute agreement
        exact_match = np.mean(llm_arr == human_arr)
        
        # Within 1 point
        within_one = np.mean(np.abs(llm_arr - human_arr) <= 1)
        
        # Correlation
        if len(llm_arr) > 1:
            correlation = np.corrcoef(llm_arr, human_arr)[0, 1]
        else:
            correlation = 0.0
        
        agreement[dim] = {
            "exact_match": round(exact_match * 100, 1),
            "within_one": round(within_one * 100, 1),
            "correlation": round(correlation, 3),
            "count": len(llm_arr),
        }
    
    # Overall
    all_llm = []
    all_human = []
    for s1, s2 in zip(llm_scores, human_scores):
        for dim in dims:
            if dim in s1 and dim in s2:
                all_llm.append(s1[dim])
                all_human.append(s2[dim])
    
    if all_llm:
        agreement["overall"] = {
            "exact_match": round(np.mean(np.array(all_llm) == np.array(all_human)) * 100, 1),
            "within_one": round(np.mean(np.abs(np.array(all_llm) - np.array(all_human)) <= 1) * 100, 1),
            "correlation": round(np.corrcoef(np.array(all_llm, dtype=float), np.array(all_human, dtype=float))[0, 1], 3),
            "count": len(all_llm),
        }
    
    return agreement


def display_agreement(agreement: dict):
    """Display agreement metrics."""
    console.print(Panel("[bold magenta]LLM vs Human Agreement[/]", border_style="magenta"))
    
    table = Table(title="Agreement Metrics")
    table.add_column("Dimension", style="cyan")
    table.add_column("Exact Match %", justify="right")
    table.add_column("Within 1 Point %", justify="right")
    table.add_column("Correlation", justify="right")
    table.add_column("Count", justify="right")
    
    for dim, metrics in agreement.items():
        color = "green" if metrics["exact_match"] >= 60 else "yellow" if metrics["exact_match"] >= 40 else "red"
        table.add_row(
            dim,
            f"[{color}]{metrics['exact_match']}%[/{color}]",
            f"{metrics['within_one']}%",
            f"{metrics['correlation']:.3f}",
            str(metrics["count"]),
        )
    
    console.print(table)
    
    # Cohen's Kappa approximation
    overall = agreement.get("overall", {})
    if overall:
        console.print("\n[bold]Overall Agreement:[/]")
        console.print(f"  Exact Match: {overall['exact_match']}%")
        console.print(f"  Within 1 Point: {overall['within_one']}%")
        console.print(f"  Correlation: {overall['correlation']:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Scoring Interface")
    parser.add_argument("--resume", action="store_true", help="Resume from last position")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N messages")
    parser.add_argument("--template", default=str(Path(__file__).parent.parent / "data" / "human_scoring_template_v3.json"))
    parser.add_argument("--judge-results", default=str(Path(__file__).parent.parent / "reports" / "v4_llm_judge_results.json"),
                        help="LLM judge results file to compute agreement against at the end")
    args = parser.parse_args()

    progress_path = str(Path(__file__).parent.parent / "data" / "human_scores_v3.json")
    
    # Load template
    console.print("[bold blue]Loading scoring template...[/]")
    template = load_template(args.template)
    console.print(f"[green]{len(template)} messages to score[/]\n")
    
    # Display rubric
    display_rubric()
    
    # Load or initialize progress
    if args.resume:
        progress = load_progress(progress_path)
        start_index = progress["last_completed_index"] + 1
        console.print(f"[yellow]Resuming from index {start_index}...[/]")
    else:
        progress = {"last_completed_index": -1, "scores": []}
        start_index = args.skip
    
    # Score messages
    for idx in range(start_index, len(template)):
        entry = template[idx]
        
        display_message(entry, idx, len(template))
        
        # Get scores
        scores = score_message()
        scores["tweet_id"] = entry["tweet_id"]
        progress["scores"].append(scores)
        
        # Save progress every 5 messages
        if (idx + 1) % 5 == 0:
            progress["last_completed_index"] = idx
            save_progress(progress_path, progress)
            console.print(f"\n[green]Progress saved ({idx + 1}/{len(template)})[/]")
        
        # Ask to continue
        if idx < len(template) - 1:
            response = Prompt.ask("\nContinue?", choices=["y", "n", "q"], default="y")
            if response == "q":
                progress["last_completed_index"] = idx
                save_progress(progress_path, progress)
                console.print("[yellow]Scoring paused. Run with --resume to continue.[/]")
                return
    
    # Save final progress
    progress["last_completed_index"] = len(template) - 1
    save_progress(progress_path, progress)
    
    # Compute agreement with LLM scores
    console.print("\n[bold blue]Computing agreement with LLM judge...[/]")
    
    with open(args.judge_results) as f:
        llm_data = json.load(f)
    
    llm_scores = [s for s in llm_data["scores"] if "error" not in s]
    llm_dict = {s["tweet_id"]: s for s in llm_scores}
    
    human_scores = progress["scores"]
    
    # Align LLM and human scores
    aligned_llm = []
    aligned_human = []
    for h in human_scores:
        if h["tweet_id"] in llm_dict:
            aligned_llm.append(llm_dict[h["tweet_id"]])
            aligned_human.append(h)
    
    agreement = compute_agreement(aligned_llm, aligned_human)
    display_agreement(agreement)
    
    # Save final results
    results_path = str(Path(__file__).parent.parent / "data" / "human_vs_llm_agreement.json")
    with open(results_path, "w") as f:
        json.dump({
            "human_scores": progress["scores"],
            "llm_scores": aligned_llm,
            "agreement": agreement,
            "sample_size": len(aligned_human),
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {results_path}[/]")


if __name__ == "__main__":
    main()
