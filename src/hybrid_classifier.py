"""Hybrid classifier combining semantic search with LLM.

Uses BGE-M3 embeddings + FAISS for semantic search to find similar
labeled messages, then either:
1. Uses majority voting (if neighbors agree)
2. Falls back to LLM with few-shot examples (if neighbors disagree)

This approach addresses the "too cautious" problem where LLM defaults
to "unclear" by providing concrete similar examples.

Usage:
    classifier = HybridClassifier()
    result = classifier.classify("My battery is draining fast")
    print(result)  # {"intent": "battery_drain_after_update", "confidence": 0.85, "method": "semantic_majority"}
"""

import json
import os
import pickle
from pathlib import Path
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_client import LLMClient


class HybridClassifier:
    """Hybrid classifier combining semantic search with LLM."""
    
    def __init__(
        self,
        index_path: str = None,
        llm_client: LLMClient = None,
        model_name: str = "BAAI/bge-m3",
        k_neighbors: int = 5,
        majority_threshold: int = 4,
        similarity_threshold: float = 0.7,
    ):
        """Initialize hybrid classifier.
        
        Args:
            index_path: Path to FAISS index (without extension)
            llm_client: LLM client for fallback classification
            model_name: Sentence-transformers model name
            k_neighbors: Number of neighbors to retrieve
            majority_threshold: Minimum neighbors agreeing for majority vote
            similarity_threshold: Minimum similarity for "likely" classification
        """
        self.index_path = index_path or str(
            Path(__file__).parent.parent / "data" / "bge_m3_index"
        )
        self.llm = llm_client or LLMClient()
        self.model_name = model_name
        self.k_neighbors = k_neighbors
        self.majority_threshold = majority_threshold
        self.similarity_threshold = similarity_threshold
        
        # Lazy-load components
        self._model = None
        self._index = None
        self._metadata = None
        self._intent_defs = None
        
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            print(f"[HYBRID] Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            
    def _load_index(self):
        """Lazy-load the FAISS index and metadata."""
        if self._index is None:
            faiss_path = f"{self.index_path}.faiss"
            pkl_path = f"{self.index_path}.pkl"
            
            if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"Index not found at {self.index_path}. "
                    f"Run build_hybrid_index.py first."
                )
            
            print(f"[HYBRID] Loading FAISS index from {self.index_path}")
            self._index = faiss.read_index(faiss_path)
            
            with open(pkl_path, "rb") as f:
                self._metadata = pickle.load(f)
            
            print(f"[HYBRID] Loaded {self._index.ntotal} vectors")
            
    def _load_intent_definitions(self):
        """Load intent definitions from JSON file."""
        if self._intent_defs is None:
            intent_path = Path(__file__).parent / "intent_definitions.json"
            with open(intent_path) as f:
                self._intent_defs = json.load(f)
        return self._intent_defs
    
    def _embed_message(self, message: str) -> np.ndarray:
        """Embed a message using BGE-M3.
        
        Args:
            message: Customer support message
            
        Returns:
            Embedding vector (1024-dimensional)
        """
        self._load_model()
        embedding = self._model.encode(
            [message],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
    
    def _find_similar(self, embedding: np.ndarray, k: int = None) -> list:
        """Find k nearest neighbors in FAISS index.
        
        Args:
            embedding: Query embedding
            k: Number of neighbors (default: self.k_neighbors)
            
        Returns:
            List of dicts with message, intent, similarity
        """
        self._load_index()
        
        if k is None:
            k = self.k_neighbors
        
        # Search
        distances, indices = self._index.search(embedding, k)
        
        # Format results
        neighbors = []
        training_data = self._metadata["training_data"]
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(training_data):
                item = training_data[idx]
                neighbors.append({
                    "text": item["text"],
                    "intent": item["labeled_intent"],
                    "similarity": float(dist),  # Inner product similarity
                    "tweet_id": item.get("tweet_id"),
                })
        
        return neighbors
    
    def _check_agreement(self, neighbors: list) -> tuple:
        """Check if neighbors agree on intent.
        
        Args:
            neighbors: List of neighbor dicts
            
        Returns:
            Tuple of (top_intent, top_count, avg_similarity)
        """
        if not neighbors:
            return ("unclear", 0, 0.0)
        
        # Count intents
        intents = [n["intent"] for n in neighbors]
        intent_counts = Counter(intents)
        top_intent, top_count = intent_counts.most_common(1)[0]
        
        # Calculate average similarity
        avg_similarity = np.mean([n["similarity"] for n in neighbors])
        
        return (top_intent, top_count, avg_similarity)
    
    def _format_few_shot(self, neighbors: list) -> str:
        """Format neighbors as few-shot examples for LLM.
        
        Args:
            neighbors: List of neighbor dicts
            
        Returns:
            Formatted string of examples
        """
        examples = []
        for i, n in enumerate(neighbors, 1):
            examples.append(
                f"Example {i}:\n"
                f"Message: \"{n['text']}\"\n"
                f"Intent: {n['intent']}\n"
                f"Similarity: {n['similarity']:.3f}"
            )
        return "\n\n".join(examples)
    
    def _classify_with_llm(self, message: str, neighbors: list) -> dict:
        """Classify using LLM with few-shot examples from neighbors.
        
        Args:
            message: Customer message to classify
            neighbors: Similar messages from semantic search
            
        Returns:
            Dict with intent, confidence, reasoning
        """
        # Build system instruction
        intent_defs = self._load_intent_definitions()
        intents = intent_defs["intents"]
        
        # Build intent descriptions
        intent_descriptions = []
        for intent in intents:
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
        
        # Build user prompt with examples
        examples_text = self._format_few_shot(neighbors)
        
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
        
        # Get classification from LLM
        response = self.llm.generate_structured(
            prompt=user_prompt,
            system_instruction=system_instruction,
        )
        
        return response
    
    def classify(self, message: str) -> dict:
        """Classify a customer message using hybrid approach.
        
        Decision logic:
        1. If 4/5 or 5/5 neighbors agree → use majority vote (no LLM)
        2. If 3/5 agree + high similarity → use majority vote (no LLM)
        3. Otherwise → use LLM with few-shot examples
        
        Args:
            message: Customer support message
            
        Returns:
            Dict with intent, confidence, method, neighbors
        """
        # Step 1: Embed message
        embedding = self._embed_message(message)
        
        # Step 2: Find similar messages
        neighbors = self._find_similar(embedding)
        
        # Step 3: Check agreement
        top_intent, top_count, avg_similarity = self._check_agreement(neighbors)
        
        # Step 4: Decision
        if top_count >= self.majority_threshold:
            # High agreement → use majority vote
            return {
                "intent": top_intent,
                "confidence": 0.85,
                "method": "semantic_majority",
                "reasoning": f"{top_count}/{self.k_neighbors} neighbors agree on {top_intent}",
                "neighbors": neighbors,
            }
        elif top_count >= 3 and avg_similarity > self.similarity_threshold:
            # Moderate agreement with good similarity → use majority vote
            return {
                "intent": top_intent,
                "confidence": 0.70,
                "method": "semantic_likely",
                "reasoning": f"{top_count}/{self.k_neighbors} neighbors agree, similarity={avg_similarity:.3f}",
                "neighbors": neighbors,
            }
        else:
            # Low agreement → use LLM with examples
            llm_result = self._classify_with_llm(message, neighbors)
            
            return {
                "intent": llm_result.get("intent", "unclear"),
                "confidence": llm_result.get("confidence", 0.5),
                "method": "llm_with_examples",
                "reasoning": llm_result.get("reasoning", ""),
                "neighbors": neighbors,
            }
    
    def classify_batch(self, messages: list) -> list:
        """Classify multiple messages.
        
        Args:
            messages: List of customer support messages
            
        Returns:
            List of classification dicts
        """
        return [self.classify(msg) for msg in messages]


if __name__ == "__main__":
    # Test the classifier
    classifier = HybridClassifier()
    
    test_messages = [
        "My battery is draining so fast after the update",
        "Can't log into my iCloud account",
        "Phone keeps crashing",
        "Safari not working",
        "@AppleSupport",
        "ios11.0.1",
    ]
    
    for msg in test_messages:
        result = classifier.classify(msg)
        print(f"\nMessage: {msg}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Method: {result['method']}")
        print(f"Reasoning: {result['reasoning']}")
