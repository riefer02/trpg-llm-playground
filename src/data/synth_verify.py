"""
Answer verification and quality scoring for synthetic data.

This module provides LLM-based verification of generated Q/A pairs
to filter out low-quality or hallucinated examples.
"""

import json
from typing import Dict, List, Optional, Tuple

from ..utils.llm_client import call_llm
from .synth_llm import WarningLimiter
from .synth_prompts import PromptConfig


def verify_qa_pair(
    context: str,
    question: str,
    answer: str,
    prompt_config: PromptConfig,
    model: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    warning_limiter: Optional[WarningLimiter] = None,
) -> Tuple[int, List[str], Optional[str]]:
    """
    Verify a Q/A pair against its source context.
    
    Returns:
        Tuple of (score, issues, corrected_answer)
        - score: 1-5 quality rating
        - issues: List of identified problems (empty if score >= 4)
        - corrected_answer: Improved answer if score < 4, else None
    """
    prompt = prompt_config.format_verification_prompt(
        context=context,
        question=question,
        answer=answer,
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
        msg = "Warning: Empty verification response from model."
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return 3, ["empty_verification_response"], None

    try:
        # Clean up response
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found")
        
        data = json.loads(clean[start:end])
        
        score = int(data.get("score", 3))
        score = max(1, min(5, score))  # Clamp to 1-5
        
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        
        corrected = data.get("corrected_answer")
        if corrected and not isinstance(corrected, str):
            corrected = None
        if corrected and not corrected.strip():
            corrected = None
            
        return score, issues, corrected
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        msg = f"Warning: Failed to parse verification response: {e}"
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return 3, ["parse_error"], None


def verify_and_filter_pairs(
    pairs: List[Dict[str, str]],
    context: str,
    prompt_config: PromptConfig,
    model: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    threshold: int = 4,
    use_corrections: bool = True,
    warning_limiter: Optional[WarningLimiter] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Verify a batch of Q/A pairs and filter by quality threshold.
    
    Args:
        pairs: List of Q/A dicts with 'instruction' and 'output' keys
        context: Source context the pairs should be grounded in
        prompt_config: Configured prompt templates
        model: LLM model to use for verification
        threshold: Minimum score to keep (1-5, default 4)
        use_corrections: If True, use corrected answers for pairs scoring < threshold
        
    Returns:
        Tuple of (filtered_pairs, stats)
        - filtered_pairs: Pairs meeting threshold (possibly with corrected answers)
        - stats: Dict with counts for each score level
    """
    filtered = []
    stats = {
        "score_5": 0,
        "score_4": 0,
        "score_3": 0,
        "score_2": 0,
        "score_1": 0,
        "corrected": 0,
        "filtered_out": 0,
    }

    for pair in pairs:
        instruction = pair.get("instruction", "")
        output = pair.get("output", "")
        
        if not instruction or not output:
            stats["filtered_out"] += 1
            continue

        score, issues, corrected = verify_qa_pair(
            context=context,
            question=instruction,
            answer=output,
            prompt_config=prompt_config,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            warning_limiter=warning_limiter,
        )

        stats[f"score_{score}"] += 1

        if score >= threshold:
            filtered.append(pair)
        elif use_corrections and corrected:
            # Use the corrected answer
            corrected_pair = dict(pair)
            corrected_pair["output"] = corrected
            corrected_pair["_was_corrected"] = True
            filtered.append(corrected_pair)
            stats["corrected"] += 1
        else:
            stats["filtered_out"] += 1

    return filtered, stats


class VerificationStats:
    """Tracks verification statistics across a generation run."""

    def __init__(self):
        self.total_verified = 0
        self.score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.corrected_count = 0
        self.filtered_count = 0

    def update(self, stats: Dict[str, int]) -> None:
        for i in range(1, 6):
            self.score_counts[i] += stats.get(f"score_{i}", 0)
            self.total_verified += stats.get(f"score_{i}", 0)
        self.corrected_count += stats.get("corrected", 0)
        self.filtered_count += stats.get("filtered_out", 0)

    def summary(self) -> Dict[str, any]:
        return {
            "total_verified": self.total_verified,
            "score_distribution": dict(self.score_counts),
            "corrected": self.corrected_count,
            "filtered_out": self.filtered_count,
            "pass_rate": (
                (self.score_counts[4] + self.score_counts[5]) / self.total_verified
                if self.total_verified > 0
                else 0.0
            ),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n--- Verification Summary ---")
        print(f"Total verified: {s['total_verified']}")
        print(f"Score distribution: {s['score_distribution']}")
        print(f"Corrected: {s['corrected']}")
        print(f"Filtered out: {s['filtered_out']}")
        print(f"Pass rate (score >= 4): {s['pass_rate']:.1%}")

