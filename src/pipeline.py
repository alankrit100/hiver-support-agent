"""CLI pipeline for the Apple Support Agent.

Input: Raw customer message (or batch of messages)
Output: {intent, confidence, drafted reply, escalate: bool, escalation_reason}

Usage:
    python pipeline.py --message "My phone keeps shutting off"
    python pipeline.py --file messages.jsonl
    python pipeline.py --interactive
"""

import argparse
import json
import time

from rich.console import Console
from rich.panel import Panel

from llm_client import LLMClient
from intent_classifier import IntentClassifier
from hybrid_classifier import HybridClassifier
from ensemble_classifier import LiveEnsembleClassifier
from retrieval import RetrievalModule
from reply_drafter import ReplyDrafter
from escalation import EscalationDecider

console = Console()


class SupportPipeline:
    """End-to-end support agent pipeline."""

    def __init__(self, build_index: bool = False, mode: str = "v4_ensemble"):
        """Initialize all pipeline components.

        Args:
            build_index: If True, rebuild the FAISS index on startup.
            mode: Classification mode - "v4_ensemble" (default, best validated
                accuracy: see reports/v4_ensemble_final_results.json),
                "llm_only", or "hybrid".
        """
        console.print("[bold blue]Initializing Apple Support Agent Pipeline...[/]")
        console.print(f"[bold blue]Mode: {mode}[/]")

        # Initialize components
        self.llm = LLMClient()
        self.mode = mode

        if mode == "v4_ensemble":
            console.print("[bold cyan]Using V4 Ensemble Classifier (BGE-M3+LR + Sarvam few-shot)[/]")
            self.classifier = LiveEnsembleClassifier(llm_client=self.llm)
        elif mode == "hybrid":
            console.print("[bold cyan]Using Hybrid Classifier (Semantic Search + LLM)[/]")
            self.classifier = HybridClassifier(llm_client=self.llm)
        else:
            console.print("[bold cyan]Using LLM-Only Classifier[/]")
            self.classifier = IntentClassifier(llm_client=self.llm)
        
        self.retrieval = RetrievalModule(max_threads=5000)
        self.drafter = ReplyDrafter(llm_client=self.llm)
        self.escalation = EscalationDecider(llm_client=self.llm)

        # Build or load index
        if build_index:
            console.print("[yellow]Building FAISS index (this may take a few minutes)...[/]")
            self.retrieval.build_index()
        else:
            if not self.retrieval._load_index():
                console.print("[yellow]No index found. Building...[/]")
                self.retrieval.build_index()

        console.print("[bold green]Pipeline ready![/]\n")

    def process_message(self, message: str) -> dict:
        """Process a single customer message through the full pipeline.

        Args:
            message: Raw customer message.

        Returns:
            Dict with full pipeline output.
        """
        start_time = time.time()

        # Step 1: Classify intent
        classification = self.classifier.classify(message)

        # Step 2: Retrieve similar threads
        retrieved = self.retrieval.retrieve(
            query=message,
            intent=classification["intent"],
            top_k=3,
        )

        # Step 3: Draft reply
        draft = self.drafter.draft(
            message=message,
            intent=classification["intent"],
            retrieved_threads=retrieved,
            confidence=classification["confidence"],
        )

        # Step 4: Escalation decision
        escalation = self.escalation.should_escalate(
            message=message,
            intent=classification["intent"],
            confidence=classification["confidence"],
            reasoning="",
            reply=draft["reply"],
            retrieved_threads=retrieved,
            grounding_source=draft["grounding_source"],
        )

        elapsed = time.time() - start_time

        retrieval_evidence = [
            {
                "tweet_id": t.get("thread_id"),
                "similarity": t.get("similarity_score"),
                "historical_response": t.get("brand_response"),
            }
            for t in retrieved
        ]

        return {
            # V4 brief required schema (Section 14):
            "intent": classification["intent"],
            "intent_confidence": classification["confidence"],
            "reply": draft["reply"],
            "escalate": escalation["escalate"],
            "escalation_reason": escalation["reason"] if escalation["escalate"] else None,
            "retrieval_evidence": retrieval_evidence,
            # Additional fields (kept for the CLI display + backward compat):
            "input_message": message,
            "confidence": classification["confidence"],
            "classification_method": classification.get("method", "llm_only"),
            "drafted_reply": draft["reply"],
            "reply_citations": draft["citations"],
            "reply_grounding": draft["grounding_source"],
            "escalation_triggers": escalation["triggers"],
            "processing_time_seconds": round(elapsed, 2),
            "token_usage": self.llm.total_tokens_used,
        }

    def process_batch(self, messages: list) -> list:
        """Process a batch of messages.

        Args:
            messages: List of raw customer messages.

        Returns:
            List of pipeline output dicts.
        """
        results = []
        for i, msg in enumerate(messages):
            console.print(f"[cyan]Processing message {i+1}/{len(messages)}...[/]")
            result = self.process_message(msg)
            results.append(result)
        return results


def display_result(result: dict):
    """Display a pipeline result in a formatted way."""
    # Intent
    intent_color = "green" if result["confidence"] > 0.7 else "yellow" if result["confidence"] > 0.4 else "red"
    console.print(Panel(
        f"[{intent_color}]{result['intent']}[/{intent_color}] (confidence: {result['confidence']:.2f})",
        title="Intent Classification",
    ))

    # Reply
    console.print(Panel(
        result["drafted_reply"],
        title="Drafted Reply",
        subtitle=f"Grounding: {result['reply_grounding']}",
    ))

    # Escalation
    if result["escalate"]:
        console.print(Panel(
            f"[bold red]ESCALATE TO HUMAN[/]\n{result['escalation_reason']}",
            title="Escalation Decision",
            border_style="red",
        ))
    else:
        console.print(Panel(
            "[green]No escalation needed[/]\nNo escalation triggers detected.",
            title="Escalation Decision",
            border_style="green",
        ))

    # Performance
    console.print(
        f"[dim]Processing time: {result['processing_time_seconds']}s | "
        f"Tokens used: {result['token_usage']['total_tokens']}[/]"
    )


def interactive_mode(pipeline: SupportPipeline):
    """Run the pipeline in interactive mode."""
    console.print("[bold green]Interactive Mode[/] (type 'quit' to exit)\n")

    while True:
        try:
            message = console.input("[bold cyan]Customer message:[/] ")
            if message.lower() in ("quit", "exit", "q"):
                break
            if not message.strip():
                continue

            result = pipeline.process_message(message)
            console.print()
            display_result(result)
            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


def main():
    parser = argparse.ArgumentParser(description="Apple Support Agent Pipeline")
    parser.add_argument("--message", "-m", type=str, help="Single message to process")
    parser.add_argument("--file", "-f", type=str, help="JSONL file with messages")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--build-index", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--output", "-o", type=str, help="Output file for results (JSONL)")
    parser.add_argument(
        "--mode",
        choices=["v4_ensemble", "llm_only", "hybrid"],
        default="v4_ensemble",
        help="Classification mode: v4_ensemble (default, validated best: "
             "68.9%% acc / 0.691 macroF1 on frozen test-45), llm_only, or hybrid"
    )
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SupportPipeline(build_index=args.build_index, mode=args.mode)

    if args.interactive:
        interactive_mode(pipeline)

    elif args.message:
        result = pipeline.process_message(args.message)
        display_result(result)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json.dumps(result) + "\n")
            console.print(f"\n[dim]Result saved to {args.output}[/]")

    elif args.file:
        with open(args.file) as f:
            messages = [json.loads(line)["message"] for line in f if line.strip()]

        results = pipeline.process_batch(messages)

        if args.output:
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            console.print(f"\n[dim]Results saved to {args.output}[/]")
        else:
            for r in results:
                display_result(r)
                console.print("\n" + "-" * 40 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
