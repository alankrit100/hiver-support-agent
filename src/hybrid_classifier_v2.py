"""Hybrid classifier V2 - LLM with semantic examples.

Always uses LLM for classification, but uses semantic search to find
similar examples that are included in the prompt. This approach:
1. Finds similar messages using BGE-M3
2. Includes them as few-shot examples in the LLM prompt
3. LLM makes the final classification decision

This addresses the issue where semantic-only approaches underperform
because semantic similarity doesn't always correlate with intent.

Usage:
    classifier = HybridClassifierV2()
    result = classifier.classify("My battery is draining fast")
    print(result)  # {"intent": "battery_drain_after_update", "confidence": 0.85, "method": "llm_with_semantic_examples"}
"""

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_client import LLMClient


class HybridClassifierV2:
    """Hybrid classifier that always uses LLM with semantic examples."""
    
    def __init__(
        self,
        index_path: str = None,
        llm_client: LLMClient = None,
        model_name: str = "BAAI/bge-m3",
        k_neighbors: int = 5,
        use_top_k: int = 3,
    ):
        """Initialize hybrid classifier V2.
        
        Args:
            index_path: Path to FAISS index (without extension)
            llm_client: LLM client for classification
            model_name: Sentence-transformers model name
            k_neighbors: Number of neighbors to retrieve
            use_top_k: Number of top neighbors to include as examples
        """
        self.index_path = index_path or str(
            Path(__file__).parent.parent / "data" / "bge_m3_index"
        )
        self.llm = llm_client or LLMClient()
        self.model_name = model_name
        self.k_neighbors = k_neighbors
        self.use_top_k = use_top_k
        
        # Lazy-load components
        self._model = None
        self._index = None
        self._metadata = None
        self._intent_defs = None
        
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            print(f"[HYBRID-V2] Loading embedding model: {self.model_name}")
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
            
            print(f"[HYBRID-V2] Loading FAISS index from {self.index_path}")
            self._index = faiss.read_index(faiss_path)
            
            with open(pkl_path, "rb") as f:
                self._metadata = pickle.load(f)
            
            print(f"[HYBRID-V2] Loaded {self._index.ntotal} vectors")
            
    def _load_intent_definitions(self):
        """Load intent definitions from JSON file."""
        if self._intent_defs is None:
            intent_path = Path(__file__).parent / "intent_definitions.json"
            with open(intent_path) as f:
                self._intent_defs = json.load(f)
        return self._intent_defs
    
    def _embed_message(self, message: str) -> np.ndarray:
        """Embed a message using BGE-M3."""
        self._load_model()
        embedding = self._model.encode(
            [message],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
    
    def _find_similar(self, embedding: np.ndarray, k: int = None) -> list:
        """Find k nearest neighbors in FAISS index."""
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
                    "similarity": float(dist),
                    "tweet_id": item.get("tweet_id"),
                })
        
        return neighbors
    
    def _format_few_shot(self, neighbors: list) -> str:
        """Format neighbors as few-shot examples for LLM."""
        # Use only top-k neighbors
        top_neighbors = neighbors[:self.use_top_k]
        
        examples = []
        for i, n in enumerate(top_neighbors, 1):
            examples.append(
                f"Example {i}:\n"
                f"Message: \"{n['text']}\"\n"
                f"Intent: {n['intent']}\n"
                f"Similarity: {n['similarity']:.3f}"
            )
        return "\n\n".join(examples)
    
    def classify(self, message: str) -> dict:
        """Classify a customer message using LLM with semantic examples.
        
        This approach:
        1. Finds similar messages using semantic search
        2. Includes them as few-shot examples in the LLM prompt
        3. LLM makes the final classification decision
        
        Args:
            message: Customer support message
            
        Returns:
            Dict with intent, confidence, method, neighbors
        """
        # Step 1: Embed message
        embedding = self._embed_message(message)
        
        # Step 2: Find similar messages
        neighbors = self._find_similar(embedding)
        
        # Step 3: Build prompt with examples
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
4. Pay attention to the similarity scores - higher similarity means more relevant examples

AVAILABLE INTENTS:
"""
        system_instruction += "\n\n".join(intent_descriptions)
        
        # Build user prompt with examples
        examples_text = self._format_few_shot(neighbors)
        
        user_prompt = f"""Classify this customer support message using the similar examples below as reference.

SIMILAR EXAMPLES (from semantic search, ordered by relevance):
{examples_text}

MESSAGE TO CLASSIFY:
"{message}"

Based on the similar examples above, classify this message. Consider both the message content and the patterns shown in the examples.

Respond in this exact JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
        
        # Step 4: Get classification from LLM
        response = self.llm.generate_structured(
            prompt=user_prompt,
            system_instruction=system_instruction,
        )
        
        return {
            "intent": response.get("intent", "unclear"),
            "confidence": response.get("confidence", 0.5),
            "method": "llm_with_semantic_examples",
            "reasoning": response.get("reasoning", ""),
            "neighbors": neighbors,
        }
    
    def classify_batch(self, messages: list) -> list:
        """Classify multiple messages."""
        return [self.classify(msg) for msg in messages]


if __name__ == "__main__":
    # Test the classifier
    classifier = HybridClassifierV2()
    
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
