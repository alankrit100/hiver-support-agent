"""Run LLM-as-Judge on pipeline-generated replies.

Scores the new replies generated with the improved prompt.
"""

import json
import os
import time
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from llm_client import LLMClient

console = Console()

# Simplified prompt for scoring
SIMPLE_PROMPT = """Rate this customer support reply on 5 dimensions (1-5 scale):

1. Groundedness: Is it based on real examples? Does it cite specific cases?
2. Relevance: Does it address the actual issue?
3. Correctness: Is the advice accurate?
4. Tone: Professional and empathetic?
5. Actionability: Clear next steps?

Customer: {message}
Reply: {reply}

Respond ONLY in JSON:
{{"groundedness": <1-5>, "relevance": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>}}"""


def load_results(path: str) -> list:
    """Load pipeline results."""
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


def score_reply(llm: LLMClient, message: str, reply: str) -> dict:
    """Score a single reply using LLM judge."""
    prompt = SIMPLE_PROMPT.format(
        message=message[:200],
        reply=reply[:200],
    )

    try:
        response = llm.generate(prompt)

        if response is None:
            return {"error": "LLM returned None"}

        # Parse JSON response
        json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            return {"error": f"Could not parse: {response[:100]}"}

    except Exception as e:
        return {"error": str(e)}


def compute_statistics(scores: list) -> dict:
    """Compute statistics from LLM judge scores."""
    dimensions = ["groundedness", "relevance", "correctness", "tone", "actionability"]
    
    stats = {}
    for dim in dimensions:
        valid_scores = [s[dim] for s in scores if dim in s and isinstance(s[dim], (int, float))]
        if valid_scores:
            stats[dim] = {
                "mean": round(sum(valid_scores) / len(valid_scores), 2),
                "min": min(valid_scores),
                "max": max(valid_scores),
                "count": len(valid_scores),
            }
        else:
            stats[dim] = {"mean": 0, "min": 0, "max": 0, "count": 0}

    # Overall score
    all_means = [stats[dim]["mean"] for dim in dimensions if stats[dim]["count"] > 0]
    stats["overall"] = round(sum(all_means) / len(all_means), 2) if all_means else 0

    return stats


def display_results(scores: list, stats: dict):
    """Display LLM judge results."""
    console.print(Panel("[bold magenta]LLM-as-Judge Results (Improved Replies)[/]", border_style="magenta"))

    # Statistics table
    table = Table(title="Score Statistics")
    table.add_column("Dimension", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Count", justify="right")

    for dim, s in stats.items():
        if dim == "overall":
            table.add_section()
            table.add_row("[bold]OVERALL[/]", f"[bold]{s}[/]", "", "", "")
        else:
            color = "green" if s["mean"] >= 4 else "yellow" if s["mean"] >= 3 else "red"
            table.add_row(
                dim,
                f"[{color}]{s['mean']}[/{color}]",
                str(s["min"]),
                str(s["max"]),
                str(s["count"]),
            )

    console.print(table)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-as-Judge on improved replies")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--results", default=str(Path(__file__).parent.parent / "data" / "pipeline_rerun_progress.json"))
    args = parser.parse_args()
    
    progress_path = str(Path(__file__).parent.parent / "data" / "judge_progress_improved.json")
    
    # Load results
    console.print("[bold blue]Loading pipeline results...[/]")
    with open(args.results) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])
    
    # Filter to only valid results (with generated_reply)
    valid_results = [r for r in results if r.get("generated_reply") and "error" not in r]
    console.print(f"[green]{len(valid_results)} replies to score[/]\n")
    
    # Initialize LLM
    llm = LLMClient()
    
    # Load or initialize progress
    if args.resume:
        progress = load_progress(progress_path)
        start_index = progress["last_completed_index"] + 1
        console.print(f"[yellow]Resuming from index {start_index}...[/]")
    else:
        progress = {"last_completed_index": -1, "scores": []}
        start_index = 0
    
    # Score replies
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Scoring replies...", total=len(valid_results))
        
        for idx, entry in enumerate(valid_results):
            if idx < start_index:
                progress_bar.update(task, advance=1)
                continue
            
            message = entry["text"]
            reply = entry["generated_reply"]
            
            scores = score_reply(llm, message, reply)
            scores["tweet_id"] = entry["tweet_id"]
            progress["scores"].append(scores)
            
            # Save progress every 5 messages
            if (idx + 1) % 5 == 0:
                progress["last_completed_index"] = idx
                save_progress(progress_path, progress)
            
            progress_bar.update(task, advance=1)
            
            # Small delay
            time.sleep(0.3)
    
    # Save final progress
    progress["last_completed_index"] = len(valid_results) - 1
    save_progress(progress_path, progress)
    
    # Compute statistics
    valid_scores = [s for s in progress["scores"] if "error" not in s]
    stats = compute_statistics(valid_scores)
    
    # Display results
    display_results(valid_scores, stats)
    
    # Save final results
    results_path = str(Path(__file__).parent.parent / "data" / "llm_judge_improved_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "scores": progress["scores"],
            "statistics": stats,
            "sample_size": len(valid_results),
            "approach": "LLM-as-Judge (pipeline-generated replies with citation enforcement)",
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {results_path}[/]")


if __name__ == "__main__":
    main()
