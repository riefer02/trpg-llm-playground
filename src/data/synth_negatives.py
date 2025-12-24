"""
Negative example generation for RAG grounding.

Generates training examples where the correct response is to acknowledge
that the context doesn't contain the answer.
"""

import json
from typing import Dict, List, Optional

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response
from .synth_llm import WarningLimiter, parse_json_list, repair_json_response
from .synth_prompts import PromptConfig


def generate_negative_pairs(
    text_chunk: str,
    prompt_config: PromptConfig,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    task_type: str,
    n_questions: int = 1,
    repair_invalid_json: bool = True,
    invalid_log_path: Optional[str] = None,
    warning_limiter: Optional[WarningLimiter] = None,
) -> List[Dict[str, str]]:
    """
    Generate Q/A pairs where the answer acknowledges the context is insufficient.
    
    These are critical for training RAG models to say "Not found in context"
    instead of hallucinating answers.
    """
    if n_questions <= 0:
        return []

    prompt = prompt_config.format_negative_prompt(
        text=text_chunk,
        n_questions=n_questions,
        task_type=task_type,
    )

    response = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
        response_format=(
            {"type": "json_object"} if "gpt-4" in model or "gpt-3.5" in model else None
        ),
    )

    if not response.strip():
        msg = "Warning: Empty negative generation response from model."
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return []

    parsed = parse_json_list(response, warning_limiter=warning_limiter)
    if parsed is not None:
        # Mark these as negative examples for tracking
        for pair in parsed:
            pair["_is_negative"] = True
        return parsed

    if invalid_log_path:
        log_invalid_response(invalid_log_path, response)

    if not repair_invalid_json:
        return []

    repaired = repair_json_response(
        response,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
        warning_limiter=warning_limiter,
    )
    if not repaired:
        return []

    repaired_parsed = parse_json_list(repaired, warning_limiter=warning_limiter)
    if repaired_parsed is None:
        if invalid_log_path:
            log_invalid_response(invalid_log_path, repaired)
        return []

    for pair in repaired_parsed:
        pair["_is_negative"] = True
    return repaired_parsed


def should_generate_negative(
    chunk_index: int,
    total_chunks: int,
    negative_ratio: float,
    seed: int = 1337,
) -> bool:
    """
    Deterministic check for whether to generate negatives for this chunk.
    
    Uses a simple hash-based approach to ensure consistent behavior across
    runs while achieving the target ratio.
    """
    import hashlib
    
    # Create deterministic hash from chunk index and seed
    hash_input = f"{seed}:{chunk_index}".encode()
    hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    threshold = int(negative_ratio * 0xFFFFFFFF)
    
    return hash_val < threshold


def calculate_negative_count(
    current_positive_count: int,
    current_negative_count: int,
    target_ratio: float,
    max_per_chunk: int = 2,
) -> int:
    """
    Calculate how many negative examples to generate to maintain target ratio.
    
    Returns 0 if we're already at or above the target ratio.
    """
    if current_positive_count == 0:
        return 0
    
    total = current_positive_count + current_negative_count
    current_ratio = current_negative_count / total if total > 0 else 0
    
    if current_ratio >= target_ratio:
        return 0
    
    # Calculate how many negatives needed to reach target
    # target_ratio = (neg + n) / (total + n)
    # Solving: n = (target_ratio * total - neg) / (1 - target_ratio)
    needed = (target_ratio * total - current_negative_count) / (1 - target_ratio)
    needed = max(0, int(needed) + 1)  # Round up
    
    return min(needed, max_per_chunk)

