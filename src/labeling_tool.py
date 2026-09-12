"""Golden set labeling tool for human annotation.

Interactive CLI that displays messages one at a time and collects
human labels for intent, escalation, and ideal reply notes.

Usage:
    python src/labeling_tool.py                              # Basic mode (no AI suggestions)
    python src/labeling_tool.py --show-ai                    # Show AI suggestions after submit
    python src/labeling_tool.py --output golden_eval_set.jsonl
    python src/labeling_tool.py --resume                     # Resume from where you left off
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

# Intent definitions
INTENTS = {
    "1": "ios_update_issues",
    "2": "battery_drain_after_update",
    "3": "device_crash_freeze",
    "4": "app_malfunction",
    "5": "account_access",
    "6": "unclear",
}


def load_candidates(path: str) -> list:
    """Load eval candidates from JSONL."""
    candidates = []
    with open(path) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def load_existing_labels(path: str) -> set:
    """Load already-labeled tweet_ids for resume support."""
    labeled_ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    labeled_ids.add(entry["tweet_id"])
    return labeled_ids


def display_message(candidate: dict, index: int, total: int):
    """Display a message with thread context."""
    # Header
    console.print()
    console.print(Panel(
        f"[bold]Message {index + 1} of {total}[/bold]",
        border_style="blue",
    ))

    # Thread context
    context = candidate.get("thread_context", [])
    if context:
        console.print("\n[bold cyan]THREAD CONTEXT:[/bold cyan]")
        for msg in context:
            author = msg["author"]
            text = msg["text"][:200]
            if len(msg["text"]) > 200:
                text += "..."
            if author == "AppleSupport":
                console.print(f"  [green]AppleSupport:[/green] {text}")
            else:
                console.print(f"  [yellow]Customer:[/yellow] {text}")

    # Target message
    console.print("\n[bold magenta]TARGET MESSAGE:[/bold magenta]")
    text = candidate["text"]
    if len(text) > 500:
        text = text[:500] + "..."
    console.print(f"  {text}")

    # Brand response if available
    brand_response = candidate.get("brand_response")
    if brand_response:
        console.print("\n[bold green]APPLE'S RESPONSE (if any):[/bold green]")
        resp = brand_response[:300]
        if len(brand_response) > 300:
            resp += "..."
        console.print(f"  {resp}")

    # Edge cases
    edge_cases = candidate.get("edge_cases", [])
    if edge_cases:
        console.print(f"\n[bold red]EDGE CASES: {', '.join(edge_cases)}[/bold red]")


def display_intent_menu():
    """Display the intent selection menu."""
    console.print("\n[bold]LABELING:[/bold]")
    console.print("  Intent (1-6):")
    for num, intent in INTENTS.items():
        console.print(f"    {num}. {intent}")


def get_label_input(candidate: dict, show_ai: bool = False, pipeline=None) -> dict:
    """Get label input from user."""
    display_intent_menu()

    # Get intent
    while True:
        intent_input = console.input("\n  Enter intent number (1-6): ").strip()
        if intent_input in INTENTS:
            labeled_intent = INTENTS[intent_input]
            break
        elif intent_input.lower() == "q":
            return None
        elif intent_input.lower() == "s":
            return {"skipped": True}
        else:
            console.print("[red]Invalid input. Enter 1-6.[/]")

    # Get escalation
    while True:
        escalate_input = console.input("  Escalate? (y/n): ").strip().lower()
        if escalate_input in ("y", "n"):
            labeled_escalate = escalate_input == "y"
            break
        elif escalate_input == "q":
            return None
        elif escalate_input == "s":
            return {"skipped": True}
        else:
            console.print("[red]Invalid input. Enter y or n.[/]")

    # Get ideal reply note
    ideal_reply = console.input("  Ideal reply note (Enter to skip): ").strip()
    if ideal_reply.lower() == "q":
        return None
    if ideal_reply.lower() == "s":
        return {"skipped": True}

    return {
        "labeled_intent": labeled_intent,
        "labeled_escalate": labeled_escalate,
        "ideal_reply_note": ideal_reply if ideal_reply else None,
    }


def show_ai_suggestion(candidate: dict, pipeline):
    """Show AI suggestion after human has submitted their label."""
    try:
        result = pipeline.process_message(candidate["text"])

        console.print(Panel(
            f"[bold]AI SUGGESTION (for comparison):[/bold]\n"
            f"Intent: {result['intent']} ({result['confidence']:.2f})\n"
            f"Escalate: {'Yes' if result['escalate'] else 'No'}\n"
            f"Reason: {result.get('escalation_reason', 'N/A')}\n"
            f"Draft: {result['drafted_reply'][:200]}...",
            title="AI Suggestion",
            border_style="yellow",
        ))

        return {
            "ai_suggested_intent": result["intent"],
            "ai_suggested_escalate": result["escalate"],
            "ai_suggested_reply": result["drafted_reply"],
        }
    except Exception as e:
        console.print(f"[red]AI suggestion failed: {e}[/]")
        return {
            "ai_suggested_intent": None,
            "ai_suggested_escalate": None,
            "ai_suggested_reply": None,
        }


def save_label(output_path: str, candidate: dict, labels: dict, ai_suggestion: dict = None):
    """Save a labeled entry to JSONL."""
    entry = {
        "tweet_id": candidate["tweet_id"],
        "text": candidate["text"],
        "thread_context": candidate.get("thread_context", []),
        "brand_response": candidate.get("brand_response"),
        "labeled_intent": labels.get("labeled_intent"),
        "labeled_escalate": labels.get("labeled_escalate"),
        "ideal_reply_note": labels.get("ideal_reply_note"),
        "ai_suggested_intent": ai_suggestion.get("ai_suggested_intent") if ai_suggestion else None,
        "ai_suggested_escalate": ai_suggestion.get("ai_suggested_escalate") if ai_suggestion else None,
        "ai_suggested_reply": ai_suggestion.get("ai_suggested_reply") if ai_suggestion else None,
        "labeled_by": "human",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detected_intent": candidate.get("detected_intent"),
        "edge_cases": candidate.get("edge_cases", []),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def display_progress(labeled: int, total: int, skipped: int):
    """Display progress bar."""
    pct = (labeled / total * 100) if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * labeled / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    console.print(
        f"\n[dim]Progress: [{bar}] {pct:.0f}% ({labeled}/{total} labeled, {skipped} skipped)[/dim]"
    )


def run_labeling_tool(args):
    """Main labeling tool loop."""
    # Load candidates
    console.print("[bold blue]Loading eval candidates...[/]")
    candidates = load_candidates(args.candidates)
    console.print(f"[green]Loaded {len(candidates)} candidates[/]")

    # Load existing labels for resume
    labeled_ids = load_existing_labels(args.output)
    if labeled_ids:
        console.print(f"[yellow]Found {len(labeled_ids)} already-labeled messages (resume mode)[/]")

    # Filter out already-labeled
    unlabeled = [c for c in candidates if c["tweet_id"] not in labeled_ids]
    if not unlabeled:
        console.print("[green]All messages already labeled![/]")
        return

    console.print(f"[cyan]Messages to label: {len(unlabeled)}[/]")

    # Initialize pipeline if showing AI suggestions
    pipeline = None
    if args.show_ai:
        console.print("[yellow]Initializing pipeline for AI suggestions...[/]")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from pipeline import SupportPipeline
            pipeline = SupportPipeline()
            console.print("[green]Pipeline ready for AI suggestions[/]")
        except Exception as e:
            console.print(f"[red]Failed to initialize pipeline: {e}[/]")
            console.print("[yellow]Continuing without AI suggestions[/]")

    # Labeling loop
    skipped = 0
    labeled = len(labeled_ids)
    total = len(candidates)
    i = 0

    console.print("\n[bold green]LABELING TOOL READY[/]")
    console.print("[dim]Controls: q=quit, s=skip, b=back, 1-6=intent, y/n=escalate[/dim]\n")

    while i < len(unlabeled):
        candidate = unlabeled[i]
        display_message(candidate, labeled, total)

        # Get labels
        labels = get_label_input(candidate, args.show_ai, pipeline)

        if labels is None:
            # Quit
            console.print(f"\n[yellow]Quitting. Progress saved ({labeled} labeled).[/]")
            break
        elif labels.get("skipped"):
            # Skip
            skipped += 1
            console.print("[dim]Skipped[/]")
            i += 1
            continue

        # Show AI suggestion if enabled
        ai_suggestion = None
        if args.show_ai and pipeline:
            ai_suggestion = show_ai_suggestion(candidate, pipeline)

        # Confirm
        console.print(f"\n[bold]Label: intent={labels['labeled_intent']}, "
                       f"escalate={labels['labeled_escalate']}[/]")
        confirm = console.input("  Confirm? (y/n/edit/q): ").strip().lower()

        if confirm == "q":
            console.print(f"\n[yellow]Quitting. Progress saved ({labeled} labeled).[/]")
            break
        elif confirm == "n":
            console.print("[dim]Re-enter labels...[/]")
            continue
        elif confirm == "edit":
            # Re-enter
            labels = get_label_input(candidate, args.show_ai, pipeline)
            if labels is None or labels.get("skipped"):
                i += 1
                continue

        # Save
        save_label(args.output, candidate, labels, ai_suggestion)
        labeled += 1
        console.print("[green]Saved![/]")

        display_progress(labeled, total, skipped)
        i += 1

    # Final summary
    console.print("\n[bold]Session complete:[/]")
    console.print(f"  Labeled: {labeled}")
    console.print(f"  Skipped: {skipped}")
    console.print(f"  Output: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Golden set labeling tool")
    parser.add_argument(
        "--candidates",
        default=str(Path(__file__).parent.parent / "data" / "eval_candidates.jsonl"),
        help="Path to eval_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "golden_eval_set.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--show-ai", action="store_true", help="Show AI suggestions after submit")
    parser.add_argument("--resume", action="store_true", help="Resume from previous session")
    args = parser.parse_args()

    run_labeling_tool(args)


if __name__ == "__main__":
    main()
