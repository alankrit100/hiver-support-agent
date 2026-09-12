"""Fast LLM-as-Judge for reply quality evaluation.

Processes replies in batches with progress saving.

Usage:
    python llm_judge_fast.py              # Score all 40 samples
    python llm_judge_fast.py --resume     # Resume from last checkpoint
"""

import json
import os
import time
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from llm_client import LLMClient

console = Console()

RUBRIC = """You are evaluating the quality of a customer support reply.

Rate the reply on each dimension from 1 (poor) to 5 (excellent).

**Dimensions:**
1. **Groundedness** (1-5): Is the reply based on real historical examples? Does it cite specific solutions?
   - 1: Completely generic, no grounding
   - 3: Somewhat grounded, references general solutions
   - 5: Highly grounded, cites specific examples

2. **Relevance** (1-5): Does the reply address the actual customer issue?
   - 1: Completely off-topic
   - 3: Partially addresses the issue
   - 5: Directly addresses the core problem

3. **Correctness** (1-5): Is the advice technically accurate?
   - 1: Incorrect advice
   - 3: Partially correct
   - 5: Fully accurate and helpful

4. **Tone** (1-5): Is the reply professional, empathetic, and appropriate?
   - 1: Robotic or rude
   - 3: Professional but generic
   - 5: Professional, empathetic, personalized

5. **Actionability** (1-5): Does the reply give clear next steps?
   - 1: No actionable guidance
   - 3: Vague guidance
   - 5: Clear, specific steps

**Input:**
Customer Message: {message}
Intent: {intent}
Drafted Reply: {reply}

**Output Format (JSON only):**
{{
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "correctness": <1-5>,
  "tone": <1-5>,
  "actionability": <1-5>,
  "reasoning": "<brief explanation>"
}}"""


def load_sample(path: str) -> list:
    """Load the 40 sample messages."""
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


def score_reply(llm: LLMClient, message: str, intent: str, reply: str) -> dict:
    """Score a single reply using LLM judge."""
    prompt = RUBRIC.format(
        message=message,
        intent=intent,
        reply=reply,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.generate, prompt)
            response = future.result(timeout=30)

        if response is None:
            return {"error": "LLM returned None"}

        # Parse JSON response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            return {"error": "Could not parse LLM response"}

    except TimeoutError:
        return {"error": "LLM call timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


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
                "median": sorted(valid_scores)[len(valid_scores) // 2],
            }
        else:
            stats[dim] = {"mean": 0, "min": 0, "max": 0, "median": 0}

    # Overall score
    all_means = [stats[dim]["mean"] for dim in dimensions]
    stats["overall"] = round(sum(all_means) / len(all_means), 2) if all_means else 0

    return stats


def display_results(scores: list, stats: dict):
    """Display LLM judge results."""
    console.print(Panel("[bold magenta]LLM-as-Judge Results[/]", border_style="magenta"))

    # Statistics table
    table = Table(title="Score Statistics")
    table.add_column("Dimension", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Median", justify="right")

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
                str(s["median"]),
            )

    console.print(table)


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Fast LLM-as-Judge")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--sample", default=str(Path(__file__).parent.parent / "data" / "judge_sample.json"))
    args = parser.parse_args()
    
    progress_path = str(Path(__file__).parent.parent / "data" / "judge_progress.json")
    
    # Load sample
    console.print("[bold blue]Loading sample...[/]")
    sample = load_sample(args.sample)
    console.print(f"[green]{len(sample)} replies to score[/]\n")
    
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
        task = progress_bar.add_task("Scoring replies...", total=len(sample))
        
        for idx, entry in enumerate(sample):
            if idx < start_index:
                progress_bar.update(task, advance=1)
                continue
            
            message = entry["text"]
            intent = entry["labeled_intent"]
            reply = entry["brand_response"]
            
            if not reply:
                scores = {"error": "No reply"}
            else:
                scores = score_reply(llm, message, intent, reply)
            
            scores["tweet_id"] = entry["tweet_id"]
            progress["scores"].append(scores)
            
            # Save progress every 5 messages
            if (idx + 1) % 5 == 0:
                progress["last_completed_index"] = idx
                save_progress(progress_path, progress)
            
            progress_bar.update(task, advance=1)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Save final progress
    progress["last_completed_index"] = len(sample) - 1
    save_progress(progress_path, progress)
    
    # Compute statistics
    valid_scores = [s for s in progress["scores"] if "error" not in s]
    stats = compute_statistics(valid_scores)
    
    # Display results
    display_results(valid_scores, stats)
    
    # Save final results
    results_path = str(Path(__file__).parent.parent / "data" / "llm_judge_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "scores": progress["scores"],
            "statistics": stats,
            "sample_size": len(sample),
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {results_path}[/]")


if __name__ == "__main__":
    main()
