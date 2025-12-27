"""
Semantic deduplication for synthetic Q/A pairs.

Uses embedding-based similarity to filter out redundant questions,
ensuring diversity in the training dataset.
"""

import hashlib
from typing import Dict, List, Tuple

# Lazy import for sentence-transformers (optional dependency)
_model_cache: Dict[str, any] = {}


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy load the sentence transformer model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer

            _model_cache[model_name] = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for deduplication. "
                "Install via: pip install sentence-transformers"
            )
    return _model_cache[model_name]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def deduplicate_pairs(
    pairs: List[Dict[str, str]],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    key: str = "instruction",
) -> Tuple[List[Dict[str, str]], int]:
    """
    Remove semantically similar pairs based on embedding similarity.

    Args:
        pairs: List of Q/A dicts
        threshold: Similarity threshold above which pairs are considered duplicates (0-1)
        model_name: Sentence transformer model to use
        key: Which field to compare for similarity (default: instruction)

    Returns:
        Tuple of (deduplicated_pairs, removed_count)
    """
    if not pairs:
        return [], 0

    if len(pairs) == 1:
        return pairs, 0

    model = _get_embedding_model(model_name)

    # Extract texts for embedding
    texts = [p.get(key, "") for p in pairs]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Greedy deduplication - keep first occurrence, remove similar subsequent
    keep_indices = []
    kept_embeddings = []

    for i, emb in enumerate(embeddings):
        is_duplicate = False
        for kept_emb in kept_embeddings:
            sim = cosine_similarity(emb.tolist(), kept_emb.tolist())
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(i)
            kept_embeddings.append(emb)

    deduplicated = [pairs[i] for i in keep_indices]
    removed = len(pairs) - len(deduplicated)

    return deduplicated, removed


def exact_hash_dedup(
    pairs: List[Dict[str, str]],
    key: str = "instruction",
) -> Tuple[List[Dict[str, str]], int]:
    """
    Remove exact duplicate questions (case-insensitive, whitespace-normalized).

    This is a fast pre-filter before semantic deduplication.
    """
    seen = set()
    deduplicated = []

    for pair in pairs:
        text = pair.get(key, "")
        # Normalize: lowercase, collapse whitespace
        normalized = " ".join(text.lower().split())
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()

        if text_hash not in seen:
            seen.add(text_hash)
            deduplicated.append(pair)

    removed = len(pairs) - len(deduplicated)
    return deduplicated, removed


class DeduplicationStats:
    """Tracks deduplication statistics across a generation run."""

    def __init__(self):
        self.total_input = 0
        self.exact_removed = 0
        self.semantic_removed = 0

    def update(
        self, input_count: int, exact_removed: int, semantic_removed: int
    ) -> None:
        self.total_input += input_count
        self.exact_removed += exact_removed
        self.semantic_removed += semantic_removed

    @property
    def total_removed(self) -> int:
        return self.exact_removed + self.semantic_removed

    @property
    def total_kept(self) -> int:
        return self.total_input - self.total_removed

    def summary(self) -> Dict[str, any]:
        return {
            "total_input": self.total_input,
            "exact_removed": self.exact_removed,
            "semantic_removed": self.semantic_removed,
            "total_removed": self.total_removed,
            "total_kept": self.total_kept,
            "dedup_rate": (
                self.total_removed / self.total_input if self.total_input > 0 else 0.0
            ),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n--- Deduplication Summary ---")
        print(f"Total input: {s['total_input']}")
        print(f"Exact duplicates removed: {s['exact_removed']}")
        print(f"Semantic duplicates removed: {s['semantic_removed']}")
        print(f"Total kept: {s['total_kept']}")
        print(f"Deduplication rate: {s['dedup_rate']:.1%}")


class RunningDeduplicator:
    """
    Maintains embedding state across chunks for running deduplication.

    This allows deduplication against ALL previously generated questions,
    not just within a single chunk.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
        key: str = "instruction",
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.key = key
        self._model = None
        self._seen_embeddings: List[List[float]] = []
        self._seen_hashes: set = set()
        self.stats = DeduplicationStats()

    def _get_model(self):
        if self._model is None:
            self._model = _get_embedding_model(self.model_name)
        return self._model

    def _normalize_hash(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def add_and_filter(self, pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter new pairs against all previously seen pairs.

        Returns only the pairs that are sufficiently unique.
        Updates internal state with the kept pairs.
        """
        if not pairs:
            return []

        input_count = len(pairs)

        # Phase 1: Exact hash dedup against seen
        exact_filtered = []
        exact_removed = 0
        for pair in pairs:
            text = pair.get(self.key, "")
            h = self._normalize_hash(text)
            if h not in self._seen_hashes:
                exact_filtered.append(pair)
            else:
                exact_removed += 1

        # Phase 2: Semantic dedup against seen embeddings
        if not exact_filtered:
            self.stats.update(input_count, exact_removed, 0)
            return []

        model = self._get_model()
        texts = [p.get(self.key, "") for p in exact_filtered]
        new_embeddings = model.encode(texts, convert_to_numpy=True)

        semantic_filtered = []
        semantic_removed = 0

        for i, emb in enumerate(new_embeddings):
            is_duplicate = False
            emb_list = emb.tolist()

            # Check against all previously seen
            for seen_emb in self._seen_embeddings:
                sim = cosine_similarity(emb_list, seen_emb)
                if sim >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                semantic_filtered.append(exact_filtered[i])
                # Add to seen state
                self._seen_embeddings.append(emb_list)
                self._seen_hashes.add(self._normalize_hash(texts[i]))
            else:
                semantic_removed += 1

        self.stats.update(input_count, exact_removed, semantic_removed)
        return semantic_filtered
