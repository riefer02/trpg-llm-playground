"""
Topic-based chunk filtering for focused synthetic data generation.

Supports three filtering modes:
1. section_keywords: Match keywords in section_path/heading
2. text_keywords: Match keywords in chunk text
3. semantic: Use embeddings to find topically relevant chunks
"""

import re
from typing import Dict, List, Optional


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def matches_keywords(text: str, keywords: List[str], match_all: bool = False) -> bool:
    """Check if text contains any/all of the keywords (case-insensitive)."""
    if not keywords:
        return True
    normalized = normalize_text(text)
    matches = [kw.lower() in normalized for kw in keywords]
    return all(matches) if match_all else any(matches)


def filter_chunks_by_section(
    chunks: List[Dict],
    section_keywords: List[str],
    match_all: bool = False,
) -> List[Dict]:
    """Filter chunks where section_path or heading contains keywords."""
    if not section_keywords:
        return chunks

    filtered = []
    for chunk in chunks:
        section_path = chunk.get("section_path", [])
        heading = chunk.get("heading", "")

        # Build combined section text
        section_text = " ".join(str(s) for s in section_path) + " " + heading

        if matches_keywords(section_text, section_keywords, match_all):
            filtered.append(chunk)

    return filtered


def filter_chunks_by_text(
    chunks: List[Dict],
    text_keywords: List[str],
    match_all: bool = False,
) -> List[Dict]:
    """Filter chunks where text content contains keywords."""
    if not text_keywords:
        return chunks

    filtered = []
    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("text_prefixed", "")
        if matches_keywords(text, text_keywords, match_all):
            filtered.append(chunk)

    return filtered


def filter_chunks_semantic(
    chunks: List[Dict],
    topic_query: str,
    threshold: float = 0.3,
    top_k: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """
    Filter chunks by semantic similarity to a topic query.

    Args:
        chunks: List of chunk dicts with 'text' field
        topic_query: Natural language description of desired topic
        threshold: Minimum cosine similarity (0-1) to include chunk
        top_k: If set, return only top K most similar chunks
        model_name: Sentence transformer model to use

    Returns:
        Filtered list of chunks sorted by relevance (most relevant first)
    """
    if not topic_query:
        return chunks

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print(
            "Warning: sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        print("Falling back to no semantic filtering.")
        return chunks

    model = SentenceTransformer(model_name)

    # Get chunk texts
    texts = [c.get("text_prefixed") or c.get("text", "") for c in chunks]

    # Encode query and chunks
    query_emb = model.encode([topic_query], normalize_embeddings=True)[0]
    chunk_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Compute cosine similarities
    similarities = np.dot(chunk_embs, query_emb)

    # Filter and sort
    scored = [(chunks[i], similarities[i]) for i in range(len(chunks))]
    scored = [(c, s) for c, s in scored if s >= threshold]
    scored.sort(key=lambda x: x[1], reverse=True)

    if top_k:
        scored = scored[:top_k]

    return [c for c, _ in scored]


def apply_topic_filter(
    chunks: List[Dict],
    filter_config: Optional[Dict],
) -> List[Dict]:
    """
    Apply topic filtering based on config.

    Config structure:
        topic_filter:
          enabled: true
          section_keywords: ["character", "pilot", "creation"]
          text_keywords: []
          match_all: false
          semantic:
            enabled: false
            query: "How to create a character and pilot in Lancer"
            threshold: 0.3
            top_k: 50
            model: "all-MiniLM-L6-v2"
    """
    if not filter_config or not filter_config.get("enabled", False):
        return chunks

    result = chunks

    # Apply section keyword filter
    section_kw = filter_config.get("section_keywords", [])
    if section_kw:
        match_all = filter_config.get("match_all", False)
        result = filter_chunks_by_section(result, section_kw, match_all)
        print(f"After section filter ({len(section_kw)} keywords): {len(result)} chunks")

    # Apply text keyword filter
    text_kw = filter_config.get("text_keywords", [])
    if text_kw:
        match_all = filter_config.get("match_all", False)
        result = filter_chunks_by_text(result, text_kw, match_all)
        print(f"After text filter ({len(text_kw)} keywords): {len(result)} chunks")

    # Apply semantic filter
    semantic_cfg = filter_config.get("semantic", {}) or {}
    if semantic_cfg.get("enabled", False):
        query = semantic_cfg.get("query", "")
        threshold = semantic_cfg.get("threshold", 0.3)
        top_k = semantic_cfg.get("top_k")
        model = semantic_cfg.get("model", "all-MiniLM-L6-v2")

        result = filter_chunks_semantic(result, query, threshold, top_k, model)
        print(f"After semantic filter (threshold={threshold}): {len(result)} chunks")

    return result

