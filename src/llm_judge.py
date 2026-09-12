"""LLM-as-Judge for reply quality evaluation.

Uses the LLM to score drafted replies on multiple dimensions:
- Groundedness: Is the reply based on retrieved history?
- Relevance: Does it address the actual issue?
- Correctness: Is the advice accurate?
- Tone: Professional, empathetic, not robotic?
- Actionability: Does it give a clear next step?

Usage:
    python llm_judge.py                    # Score all replies
    python llm_judge.py --sample 40        # Score random sample of 40
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List
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
}}
"""


class LLMJudge:
    """Uses LLM to score reply quality."""

    def __init__(self):
        self.llm = LLMClient()
        self.scores = []

    def score_reply(self, message: str, intent: str, reply: str) -> Dict:
        """Score a single reply using LLM judge."""
        prompt = RUBRIC.format(
            message=message,
            intent=intent,
            reply=reply,
        )

        # Call LLM with timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.llm.generate, prompt)
                response = future.result(timeout=30)

            if response is None:
                return {"error": "LLM returned None"}

            # Parse JSON response
            # Try to extract JSON from response
            import re
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

    def score_batch(self, entries: List[dict], max_workers: int = 1) -> List[Dict]:
        """Score a batch of replies."""
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scoring replies...", total=len(entries))

            for entry in entries:
                message = entry["text"]
                intent = entry["labeled_intent"]
                reply = entry.get("brand_response", "")

                if not reply:
                    results.append({"error": "No reply to score"})
                    progress.update(task, advance=1)
                    continue

                scores = self.score_reply(message, intent, reply)
                results.append(scores)

                # Small delay to avoid rate limiting
                time.sleep(0.5)

                progress.update(task, advance=1)

        return results


def compute_score_statistics(scores: List[Dict]) -> Dict:
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


def display_judge_results(scores: List[Dict], stats: Dict):
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

    # Sample scores
    console.print("\n[bold]Sample Scores:[/]")
    for i, score in enumerate(scores[:5]):
        if "error" in score:
            console.print(f"  {i+1}. [red]Error: {score['error']}[/]")
        else:
            console.print(f"  {i+1}. Groundedness: {score.get('groundedness', 'N/A')}, "
                          f"Relevance: {score.get('relevance', 'N/A')}, "
                          f"Correctness: {score.get('correctness', 'N/A')}, "
                          f"Tone: {score.get('tone', 'N/A')}, "
                          f"Actionability: {score.get('actionability', 'N/A')}")


def run_llm_judge(golden_set: List[dict], sample_size: int = None) -> Dict:
    """Run LLM judge on golden set."""
    # Sample if needed
    if sample_size and sample_size < len(golden_set):
        entries = random.sample(golden_set, sample_size)
        console.print(f"[yellow]Sampled {sample_size} replies for judging[/]")
    else:
        entries = golden_set
        console.print(f"[green]Scoring all {len(entries)} replies[/]")

    # Initialize judge
    judge = LLMJudge()

    # Score replies
    scores = judge.score_batch(entries)

    # Compute statistics
    stats = compute_score_statistics(scores)

    # Display results
    display_judge_results(scores, stats)

    return {
        "scores": scores,
        "statistics": stats,
        "sample_size": len(entries),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge for reply quality")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for judging")
    args = parser.parse_args()

    # Load golden set
    golden_set_path = str(Path(__file__).parent.parent / "data" / "golden_eval_set.jsonl")

    golden_set = []
    with open(golden_set_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("labeled_by") == "human" and entry.get("brand_response"):
                    golden_set.append(entry)

    console.print(f"Loaded {len(golden_set)} labeled messages with replies\n")
    run_llm_judge(golden_set, sample_size=args.sample)
