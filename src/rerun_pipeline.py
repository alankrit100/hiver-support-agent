"""Re-run pipeline on sample messages with improved reply drafter.

Generates new replies using the updated prompt with citation enforcement.
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from pipeline import SupportPipeline

console = Console()


def load_sample(path: str) -> list:
    """Load the 40 sample messages."""
    with open(path) as f:
        return json.load(f)


def generate_reply(pipeline: SupportPipeline, entry: dict) -> dict:
    """Generate a reply for a single message."""
    try:
        result = pipeline.process_message(entry["text"])
        return {
            "tweet_id": entry["tweet_id"],
            "text": entry["text"],
            "labeled_intent": entry["labeled_intent"],
            "brand_response": entry.get("brand_response", ""),
            "generated_reply": result.get("drafted_reply", ""),
            "detected_intent": result.get("intent", ""),
            "confidence": result.get("confidence", 0),
            "citations": result.get("reply_citations", []),
            "grounding": result.get("reply_grounding", ""),
        }
    except Exception as e:
        return {
            "tweet_id": entry["tweet_id"],
            "text": entry["text"],
            "labeled_intent": entry["labeled_intent"],
            "brand_response": entry.get("brand_response", ""),
            "generated_reply": "",
            "error": str(e),
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-run pipeline on samples")
    parser.add_argument("--sample", default=str(Path(__file__).parent.parent / "data" / "judge_sample.json"))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    progress_path = str(Path(__file__).parent.parent / "data" / "pipeline_rerun_progress.json")
    
    # Load sample
    console.print("[bold blue]Loading sample...[/]")
    sample = load_sample(args.sample)
    console.print(f"[green]{len(sample)} messages to process[/]\n")
    
    # Initialize pipeline
    console.print("[bold blue]Initializing pipeline...[/]")
    pipeline = SupportPipeline()
    console.print("[green]Pipeline ready[/]\n")
    
    # Load or initialize progress
    if args.resume and Path(progress_path).exists():
        with open(progress_path) as f:
            progress = json.load(f)
        start_index = progress["last_completed_index"] + 1
        console.print(f"[yellow]Resuming from index {start_index}...[/]")
    else:
        progress = {"last_completed_index": -1, "results": []}
        start_index = 0
    
    # Process messages
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Generating replies...", total=len(sample))
        
        for idx in range(start_index, len(sample)):
            entry = sample[idx]
            
            result = generate_reply(pipeline, entry)
            progress["results"].append(result)
            
            # Save progress every 5 messages
            if (idx + 1) % 5 == 0:
                progress["last_completed_index"] = idx
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
            
            progress_bar.update(task, advance=1)
            
            # Small delay
            time.sleep(0.3)
    
    # Save final progress
    progress["last_completed_index"] = len(sample) - 1
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    # Save results
    results_path = str(Path(__file__).parent.parent / "data" / "pipeline_rerun_results.json")
    with open(results_path, "w") as f:
        json.dump(progress["results"], f, indent=2)
    
    console.print(f"\n[green]Results saved to {results_path}[/]")
    
    # Display sample results
    console.print(Panel("[bold magenta]Sample Generated Replies[/]", border_style="magenta"))
    
    table = Table(title="Generated Replies")
    table.add_column("Tweet ID", style="cyan")
    table.add_column("Intent", style="green")
    table.add_column("Generated Reply", max_width=60)
    
    for r in progress["results"][:5]:
        reply = r.get("generated_reply", "")[:100] + "..." if len(r.get("generated_reply", "")) > 100 else r.get("generated_reply", "")
        table.add_row(
            str(r["tweet_id"]),
            r.get("labeled_intent", ""),
            reply,
        )
    
    console.print(table)


if __name__ == "__main__":
    main()
