"""Retrieval module for finding similar historically resolved threads.

Uses sentence-transformers for embeddings and FAISS for vector similarity search.
Filters threads based on resolution heuristics defined in intent_definitions.json.
"""

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class RetrievalModule:
    """Retrieves similar historically resolved threads using FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = None,
        max_threads: int = 10000,
    ):
        """
        Args:
            model_name: Sentence-transformers model for embeddings.
            index_path: Path to save/load FAISS index. If None, builds fresh.
            max_threads: Maximum number of threads to index.
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.threads = []  # List of thread dicts
        self.index_path = index_path or str(
            Path(__file__).parent.parent / "data" / "faiss_index"
        )
        self.max_threads = max_threads

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self.model is None:
            print(f"[RETRIEVAL] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, csv_path: str = None):
        """Build FAISS index from the dataset.

        Args:
            csv_path: Path to twcs.csv. If None, uses default path.
        """
        self._load_model()

        if csv_path is None:
            csv_path = str(Path(__file__).parent.parent / "data" / "twcs.csv")

        print(f"[RETRIEVAL] Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)

        # Filter for AppleSupport
        apple_brand = df[df["author_id"] == "AppleSupport"]
        apple_response_ids = set(apple_brand["tweet_id"].values)

        print(f"[RETRIEVAL] Finding resolved threads...")

        # Find resolved threads based on heuristics
        resolved_threads = self._find_resolved_threads(df, apple_response_ids)

        # Limit to max_threads
        if len(resolved_threads) > self.max_threads:
            print(f"[RETRIEVAL] Limiting to {self.max_threads} threads")
            resolved_threads = resolved_threads[: self.max_threads]

        self.threads = resolved_threads
        print(f"[RETRIEVAL] Found {len(self.threads)} resolved threads")

        # Create embeddings
        print(f"[RETRIEVAL] Creating embeddings...")
        thread_texts = [t["customer_query"] for t in self.threads]
        embeddings = self.model.encode(thread_texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index
        print(f"[RETRIEVAL] Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"[RETRIEVAL] Index built with {self.index.ntotal} vectors")

        # Save index and threads
        self._save_index()

    def _find_resolved_threads(self, df: pd.DataFrame, brand_response_ids: set) -> list:
        """Find resolved threads using resolution heuristics.

        Resolution criteria (from intent_definitions.json):
        1. Customer expresses gratitude AND does not report ongoing issue
        2. Customer confirms issue is fixed
        3. Brand provided actual troubleshooting AND customer stopped responding
        4. Thread has 2+ turns and brand provided solution

        Exclusions:
        - Brand response is ONLY "DM us" = NOT resolved
        - Customer expresses continued frustration = NOT resolved
        """
        resolved = []
        gratitude_keywords = ["thank", "thanks", "appreciate"]
        fix_keywords = ["working", "worked", "resolved", "got them back", "fixed"]
        dm_deflection_keywords = [
            "dm us",
            "send us a dm",
            "let us know in dm",
            "direct message",
        ]
        frustration_keywords = [
            "still",
            "doesn't work",
            "didn't work",
            "not working",
            "frustrated",
            "angry",
        ]

        # Get customer queries
        customer_queries = df[
            (df["in_response_to_tweet_id"].isin(brand_response_ids))
            & (df["inbound"] == True)
        ]

        # Pre-filter for AppleSupport responses to speed up lookups
        apple_support_df = df[df["author_id"] == "AppleSupport"]

        for _, cust_row in tqdm(
            customer_queries.iterrows(),
            total=min(len(customer_queries), self.max_threads * 20),  # Estimate for progress
            desc="Finding resolved threads",
        ):
            # Early exit if we have enough threads
            if len(resolved) >= self.max_threads:
                print(f"\n[RETRIEVAL] Found {len(resolved)} threads, stopping early")
                break

            thread_id = cust_row["tweet_id"]
            customer_query = str(cust_row["text"])

            # Find brand response (optimized: use pre-filtered df)
            brand_resp = apple_support_df[
                apple_support_df["in_response_to_tweet_id"] == thread_id
            ]

            if len(brand_resp) == 0:
                continue

            brand_response = str(brand_resp.iloc[0]["text"])
            brand_response_lower = brand_response.lower()

            # Check for DM deflection (exclusion)
            is_dm_deflection = any(
                kw in brand_response_lower for kw in dm_deflection_keywords
            )
            has_actual_troubleshooting = not is_dm_deflection and len(brand_response) > 50

            # Find customer follow-up
            cust_followup = df[
                (df["in_response_to_tweet_id"] == brand_resp.iloc[0]["tweet_id"])
                & (df["inbound"] == True)
            ]

            if len(cust_followup) == 0:
                # No follow-up: resolved only if brand provided actual troubleshooting
                if has_actual_troubleshooting:
                    resolved.append(
                        {
                            "customer_query": customer_query,
                            "brand_response": brand_response,
                            "resolution_type": "single_turn_with_troubleshooting",
                            "thread_id": thread_id,
                        }
                    )
                continue

            # Has follow-up: check if customer is satisfied
            followup_text = str(cust_followup.iloc[0]["text"]).lower()

            # Check for gratitude
            has_gratitude = any(kw in followup_text for kw in gratitude_keywords)

            # Check for fix confirmation
            has_fix = any(kw in followup_text for kw in fix_keywords)

            # Check for frustration
            has_frustration = any(kw in followup_text for kw in frustration_keywords)

            # Resolution criteria
            if has_gratitude and not has_frustration:
                resolved.append(
                    {
                        "customer_query": customer_query,
                        "brand_response": brand_response,
                        "resolution_type": "gratitude_signal",
                        "thread_id": thread_id,
                    }
                )
            elif has_fix:
                resolved.append(
                    {
                        "customer_query": customer_query,
                        "brand_response": brand_response,
                        "resolution_type": "fix_confirmed",
                        "thread_id": thread_id,
                    }
                )

        return resolved

    def retrieve(
        self, query: str, intent: str = None, top_k: int = 3, pool_multiplier: int = 5
    ) -> list:
        """Retrieve similar resolved threads, preferring the predicted intent.

        Per the V4 brief: retrieval should not be pure semantic nearest-neighbour
        across all intents. We search a wider candidate pool, then prefer
        candidates whose (offline-predicted) intent matches `intent`, falling
        back to the next-best semantic matches if too few same-intent
        candidates exist (documented cold-start fallback, not silently
        returning irrelevant-intent examples without saying so).

        Args:
            query: Customer message to find similar threads for.
            intent: Optional predicted intent to prefer.
            top_k: Number of results to return.
            pool_multiplier: How much wider than top_k to search before
                filtering by intent (ignored when intent is None).

        Returns:
            List of thread dicts with similarity_score, predicted_intent, and
            matched_intent (bool: whether predicted_intent == intent).
        """
        self._load_model()

        if self.index is None:
            # Try to load existing index
            if not self._load_index():
                raise RuntimeError(
                    "No FAISS index found. Call build_index() first."
                )

        if self.threads and "predicted_intent" not in self.threads[0]:
            self._label_threads_with_intent()
            self._save_index()

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search a wider pool when we need to filter by intent afterward.
        pool_k = min(top_k * pool_multiplier, self.index.ntotal) if intent else top_k
        distances, indices = self.index.search(query_embedding, pool_k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.threads):
                thread = self.threads[idx].copy()
                thread["similarity_score"] = float(1 / (1 + dist))  # Convert distance to similarity
                thread["matched_intent"] = bool(intent) and thread.get("predicted_intent") == intent
                candidates.append(thread)

        if not intent:
            return candidates[:top_k]

        same_intent = [c for c in candidates if c["matched_intent"]]
        other = [c for c in candidates if not c["matched_intent"]]
        # Prefer same-intent matches; pad with best remaining semantic matches
        # if the intent-constrained pool is too small (cold-start fallback).
        return (same_intent + other)[:top_k]

    def _label_threads_with_intent(self):
        """Offline-label each indexed thread with a predicted intent, so
        retrieve() can constrain candidates by intent instead of doing pure
        semantic nearest-neighbour across all intents.

        Uses the V4 bge_lr classifier (BGE-M3 embeddings + LogisticRegression,
        fit on all 170 dev examples) -- the same model selected in
        reports/v4_cv_results.json -- rather than introducing a separate
        classifier just for this. This is a one-time cost per index build/load
        and gets cached into the saved index (see retrieve()'s guard above).
        """
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).parent))
        from v4_classifier import build_bge_lr

        repo_root = Path(__file__).parent.parent
        train_path = repo_root / "data" / "train_set.jsonl"
        emb_path = repo_root / "data" / "v4_bge_m3_dev170.npy"
        ids_path = repo_root / "data" / "v4_bge_m3_dev170.ids.json"

        if not (train_path.exists() and emb_path.exists() and ids_path.exists()):
            print("[RETRIEVAL] Missing V4 dev data/embeddings; skipping intent labeling "
                  "(retrieval will fall back to pure semantic search)")
            for t in self.threads:
                t["predicted_intent"] = None
            return

        dev = [json.loads(line) for line in open(train_path) if line.strip()]
        assert json.loads(ids_path.read_text()) == [r["tweet_id"] for r in dev], \
            "dev embedding cache order mismatch vs train_set.jsonl"
        X_dev = np.load(emb_path)
        y_dev = [r["labeled_intent"] for r in dev]
        clf = build_bge_lr(C=2.0)
        clf.fit(X_dev, y_dev)

        print(f"[RETRIEVAL] Labeling {len(self.threads)} indexed threads with predicted intent...")
        from sentence_transformers import SentenceTransformer

        bge_model = SentenceTransformer("BAAI/bge-m3")
        texts = [t["customer_query"] for t in self.threads]
        thread_embeddings = bge_model.encode(texts, normalize_embeddings=True,
                                             show_progress_bar=True)
        preds = clf.predict(np.asarray(thread_embeddings, dtype=np.float32))
        for t, p in zip(self.threads, preds):
            t["predicted_intent"] = p

    def _save_index(self):
        """Save FAISS index and threads to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.threads, f)
        print(f"[RETRIEVAL] Index saved to {self.index_path}")

    def _load_index(self) -> bool:
        """Load FAISS index and threads from disk."""
        faiss_path = f"{self.index_path}.faiss"
        pkl_path = f"{self.index_path}.pkl"

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            return False

        print(f"[RETRIEVAL] Loading index from {self.index_path}")
        self.index = faiss.read_index(faiss_path)
        with open(pkl_path, "rb") as f:
            self.threads = pickle.load(f)
        print(f"[RETRIEVAL] Loaded {self.index.ntotal} vectors, {len(self.threads)} threads")
        return True


if __name__ == "__main__":
    # Test the retrieval module
    retrieval = RetrievalModule(max_threads=1000)
    retrieval.build_index()

    test_queries = [
        "My phone keeps shutting off",
        "Battery draining fast",
        "Can't log into iCloud",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieval.retrieve(query, top_k=2)
        for i, r in enumerate(results):
            print(f"  {i+1}. (score: {r['similarity_score']:.3f}) {r['customer_query'][:100]}")
            print(f"     Response: {r['brand_response'][:100]}")
