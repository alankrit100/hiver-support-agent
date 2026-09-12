"""Optimized hybrid parameter tuning.

Loads the embedding model once and tests multiple parameter combinations.

Usage:
    python tune_hybrid_optimized.py              # Run all tuning tests
    python tune_hybrid_optimized.py --quick      # Run quick test (3 combinations)
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

console = Console()


def load_test_set(path: str) -> list:
    """Load test set from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("labeled_by") == "human":
                    data.append(entry)
    return data


def load_index_and_model(index_path: str, model_name: str = "BAAI/bge-m3"):
    """Load FAISS index and embedding model once."""
    console.print(f"[bold]Loading embedding model: {model_name}[/]")
    model = SentenceTransformer(model_name)
    
    console.print(f"[bold]Loading FAISS index from {index_path}[/]")
    faiss_path = f"{index_path}.faiss"
    pkl_path = f"{index_path}.pkl"
    
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)
    
    console.print(f"[green]Loaded {index.ntotal} vectors[/]\n")
    return model, index, metadata


def classify_with_params(
    message: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: dict,
    k_neighbors: int,
    majority_threshold: int,
    similarity_threshold: float,
) -> dict:
    """Classify a single message with given parameters."""
    from collections import Counter
    from llm_client import LLMClient
    
    # Embed message
    embedding = model.encode([message], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32)
    
    # Find neighbors
    distances, indices = index.search(embedding, k_neighbors)
    
    neighbors = []
    training_data = metadata["training_data"]
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(training_data):
            item = training_data[idx]
            neighbors.append({
                "text": item["text"],
                "intent": item["labeled_intent"],
                "similarity": float(dist),
            })
    
    # Check agreement
    if not neighbors:
        return {"intent": "unclear", "method": "no_neighbors"}
    
    intents = [n["intent"] for n in neighbors]
    intent_counts = Counter(intents)
    top_intent, top_count = intent_counts.most_common(1)[0]
    avg_similarity = np.mean([n["similarity"] for n in neighbors])
    
    # Decision
    if top_count >= majority_threshold:
        return {"intent": top_intent, "method": "semantic_majority"}
    elif top_count >= 3 and avg_similarity > similarity_threshold:
        return {"intent": top_intent, "method": "semantic_likely"}
    else:
        # Use LLM with examples
        llm = LLMClient()
        
        # Build prompt
        from intent_definitions import load_intent_definitions
        intent_defs = load_intent_definitions()
        intents_list = intent_defs["intents"]
        
        intent_descriptions = []
        for intent in intents_list:
            intent_descriptions.append(
                f"**{intent['name']}**: {intent['description']}\n"
                f"  Boundary: {intent['boundary_notes']}"
            )
        
        system_instruction = """You are an Apple Support message classifier. Classify the message into one of the predefined intent categories.

RULES:
1. Choose EXACTLY ONE intent from the list below
2. Provide a confidence score between 0.0 and 1.0
3. Use the similar examples below to guide your classification

AVAILABLE INTENTS:
"""
        system_instruction += "\n\n".join(intent_descriptions)
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\nMessage: \"{n['text']}\"\nIntent: {n['intent']}\nSimilarity: {n['similarity']:.3f}"
            for i, n in enumerate(neighbors)
        ])
        
        user_prompt = f"""Classify this customer support message using the similar examples below as reference.

SIMILAR EXAMPLES (from semantic search):
{examples_text}

MESSAGE TO CLASSIFY:
"{message}"

Based on the similar examples above, classify this message.

Respond in this exact JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
        
        response = llm.generate_structured(
            prompt=user_prompt,
            system_instruction=system_instruction,
        )
        
        return {
            "intent": response.get("intent", "unclear"),
            "method": "llm_with_examples"
        }


def evaluate_classifier(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: dict,
    test_set: list,
    intents: list,
    k_neighbors: int,
    majority_threshold: int,
    similarity_threshold: float,
) -> dict:
    """Evaluate classifier on test set."""
    y_true = []
    y_pred = []
    method_counts = defaultdict(int)
    
    for entry in test_set:
        message = entry["text"]
        true_intent = entry["labeled_intent"]
        
        try:
            result = classify_with_params(
                message, model, index, metadata,
                k_neighbors, majority_threshold, similarity_threshold
            )
            pred_intent = result["intent"]
            method = result.get("method", "unknown")
            method_counts[method] += 1
        except Exception:
            pred_intent = "unclear"
            method = "error"
        
        y_true.append(true_intent)
        y_pred.append(pred_intent)
    
    # Compute accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    # Compute per-intent metrics
    per_intent = {}
    for intent in intents:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p == intent)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != intent and p == intent)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == intent and p != intent)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_intent[intent] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    
    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": len(y_true),
        "per_intent": per_intent,
        "method_distribution": dict(method_counts),
    }


def run_tuning(test_set: list, intents: list, quick: bool = False):
    """Run parameter tuning."""
    # Load model and index once
    model, index, metadata = load_index_and_model(
        str(Path(__file__).parent.parent / "data" / "bge_m3_index")
    )
    
    # Define parameter combinations to test
    param_combos = [
        # Baseline
        {"k_neighbors": 5, "majority_threshold": 4, "similarity_threshold": 0.7, "name": "baseline"},
        
        # Test 1: Stricter majority
        {"k_neighbors": 5, "majority_threshold": 5, "similarity_threshold": 0.7, "name": "strict_majority"},
        
        # Test 2: Higher similarity
        {"k_neighbors": 5, "majority_threshold": 4, "similarity_threshold": 0.9, "name": "high_similarity"},
        
        # Test 3: More neighbors + strict majority
        {"k_neighbors": 7, "majority_threshold": 6, "similarity_threshold": 0.8, "name": "more_neighbors_strict"},
        
        # Test 4: Conservative
        {"k_neighbors": 5, "majority_threshold": 5, "similarity_threshold": 0.95, "name": "conservative"},
        
        # Test 5: Balanced
        {"k_neighbors": 5, "majority_threshold": 4, "similarity_threshold": 0.85, "name": "balanced"},
        
        # Test 6: Very strict
        {"k_neighbors": 7, "majority_threshold": 7, "similarity_threshold": 0.9, "name": "very_strict"},
    ]
    
    if quick:
        param_combos = param_combos[:4]  # Only first 4 for quick test
    
    results = []
    
    console.print(Panel("[bold cyan]HYBRID PARAMETER TUNING (OPTIMIZED)[/]", border_style="cyan"))
    console.print(f"Testing {len(param_combos)} parameter combinations on {len(test_set)} messages\n")
    
    for i, params in enumerate(param_combos, 1):
        name = params.pop("name")
        console.print(f"[bold]Test {i}/{len(param_combos)}: {name}[/]")
        console.print(f"  k={params['k_neighbors']}, threshold={params['majority_threshold']}, similarity={params['similarity_threshold']}")
        
        # Evaluate
        start_time = time.time()
        eval_result = evaluate_classifier(
            model, index, metadata, test_set, intents,
            **params
        )
        elapsed = time.time() - start_time
        
        eval_result["name"] = name
        eval_result["params"] = params
        eval_result["time_seconds"] = round(elapsed, 2)
        results.append(eval_result)
        
        acc_color = "green" if eval_result["accuracy"] >= 0.65 else "yellow" if eval_result["accuracy"] >= 0.60 else "red"
        console.print(f"  [{acc_color}]Accuracy: {eval_result['accuracy']:.3f} ({eval_result['correct']}/{eval_result['total']})[/]")
        console.print(f"  Methods: {eval_result['method_distribution']}")
        console.print(f"  Time: {elapsed:.1f}s\n")
    
    return results


def display_results(results: list):
    """Display tuning results in a formatted table."""
    console.print(Panel("[bold green]TUNING RESULTS[/]", border_style="green"))
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    # Main results table
    table = Table(title="Parameter Tuning Results")
    table.add_column("Rank", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("k", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Time", justify="right")
    
    for rank, result in enumerate(sorted_results, 1):
        acc_color = "green" if result["accuracy"] >= 0.65 else "yellow" if result["accuracy"] >= 0.60 else "red"
        table.add_row(
            str(rank),
            result["name"],
            str(result["params"]["k_neighbors"]),
            str(result["params"]["majority_threshold"]),
            str(result["params"]["similarity_threshold"]),
            f"[{acc_color}]{result['accuracy']:.3f}[/{acc_color}]",
            f"{result['correct']}/{result['total']}",
            f"{result['time_seconds']:.1f}s",
        )
    
    console.print(table)
    
    # Best result details
    best = sorted_results[0]
    console.print(f"\n[bold green]BEST CONFIGURATION: {best['name']}[/]")
    console.print(f"  Accuracy: {best['accuracy']:.3f}")
    console.print(f"  Parameters: k={best['params']['k_neighbors']}, threshold={best['params']['majority_threshold']}, similarity={best['params']['similarity_threshold']}")
    console.print(f"  Method distribution: {best['method_distribution']}")
    
    # Per-intent comparison for top 3
    if len(sorted_results) >= 3:
        console.print("\n[bold]Top 3 - Per-Intent F1:[/]")
        intent_table = Table(title="F1 Comparison")
        intent_table.add_column("Intent", style="cyan")
        for i, result in enumerate(sorted_results[:3], 1):
            intent_table.add_column(f"#{i} {result['name']}", justify="right")
        
        intents = list(sorted_results[0]["per_intent"].keys())
        for intent in intents:
            row = [intent]
            for result in sorted_results[:3]:
                f1 = result["per_intent"][intent]["f1"]
                row.append(f"{f1:.3f}")
            intent_table.add_row(*row)
        
        console.print(intent_table)
    
    return sorted_results


def main():
    parser = argparse.ArgumentParser(description="Tune hybrid classifier parameters")
    parser.add_argument(
        "--test-set",
        default=str(Path(__file__).parent.parent / "data" / "test_set.jsonl"),
        help="Path to test set"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (4 combinations instead of 7)"
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "tuning_results.json"),
        help="Output file for results"
    )
    args = parser.parse_args()
    
    # Load test set
    console.print("[bold blue]Loading test set...[/]")
    test_set = load_test_set(args.test_set)
    console.print(f"[green]{len(test_set)} test messages loaded[/]\n")
    
    # Load intents
    intents_path = str(Path(__file__).parent / "intent_definitions.json")
    with open(intents_path) as f:
        intent_data = json.load(f)
    intents = [i["name"] for i in intent_data["intents"]]
    
    # Run tuning
    results = run_tuning(test_set, intents, quick=args.quick)
    
    # Display results
    display_results(results)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {args.output}[/]")


if __name__ == "__main__":
    main()
