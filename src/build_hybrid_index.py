"""Build BGE-M3 FAISS index for hybrid classifier.

This script builds a FAISS index using the BGE-M3 embedding model
on the training set (170 messages). The index is used by the hybrid
classifier to find similar labeled messages for few-shot prompting.

Usage:
    python build_hybrid_index.py                    # Build from default training set
    python build_hybrid_index.py --force            # Force rebuild
    python build_hybrid_index.py --model BAAI/bge-m3  # Use different model
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_training_data(path: str) -> list:
    """Load training set from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data.append(entry)
    return data


def build_index(
    training_path: str,
    index_path: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> dict:
    """Build FAISS index from training data.
    
    Args:
        training_path: Path to training set JSONL
        index_path: Path to save FAISS index (without extension)
        model_name: Sentence-transformers model name
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with build statistics
    """
    start_time = time.time()
    
    # Load training data
    print(f"[BUILD] Loading training data from {training_path}...")
    training_data = load_training_data(training_path)
    print(f"[BUILD] Loaded {len(training_data)} messages")
    
    # Load embedding model
    print(f"[BUILD] Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"[BUILD] Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    
    # Extract texts and labels
    texts = [item["text"] for item in training_data]
    labels = [item["labeled_intent"] for item in training_data]
    
    # Encode texts
    print(f"[BUILD] Encoding {len(texts)} messages...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"[BUILD] Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print(f"[BUILD] Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"[BUILD] Index built with {index.ntotal} vectors")
    
    # Save index and metadata
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    # Save FAISS index
    faiss_path = f"{index_path}.faiss"
    faiss.write_index(index, faiss_path)
    print(f"[BUILD] FAISS index saved to {faiss_path}")
    
    # Save metadata (training data + labels)
    metadata = {
        "training_data": training_data,
        "model_name": model_name,
        "dimension": dimension,
        "num_vectors": index.ntotal,
    }
    pkl_path = f"{index_path}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[BUILD] Metadata saved to {pkl_path}")
    
    elapsed = time.time() - start_time
    
    stats = {
        "num_messages": len(training_data),
        "num_vectors": index.ntotal,
        "dimension": dimension,
        "model_name": model_name,
        "build_time_seconds": round(elapsed, 2),
        "index_path": faiss_path,
        "metadata_path": pkl_path,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 FAISS index")
    parser.add_argument(
        "--training-data",
        default=str(Path(__file__).parent.parent / "data" / "train_set.jsonl"),
        help="Path to training set JSONL"
    )
    parser.add_argument(
        "--index-path",
        default=str(Path(__file__).parent.parent / "data" / "bge_m3_index"),
        help="Path to save index (without extension)"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Sentence-transformers model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    args = parser.parse_args()
    
    # Check if index exists
    faiss_path = f"{args.index_path}.faiss"
    if os.path.exists(faiss_path) and not args.force:
        print(f"[BUILD] Index already exists at {faiss_path}")
        print(f"[BUILD] Use --force to rebuild")
        return
    
    # Build index
    print("=" * 60)
    print("BUILDING BGE-M3 FAISS INDEX")
    print("=" * 60)
    
    stats = build_index(
        training_path=args.training_data,
        index_path=args.index_path,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Messages indexed: {stats['num_messages']}")
    print(f"Vectors: {stats['num_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Model: {stats['model_name']}")
    print(f"Build time: {stats['build_time_seconds']}s")
    print(f"Index file: {stats['index_path']}")
    print(f"Metadata file: {stats['metadata_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
