from typing import Dict, List, Optional

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response
from .synth_llm import WarningLimiter, parse_json_list, repair_json_response

COVERAGE_PROMPT_TEMPLATE = """
You are an expert RPG rules compiler. Your job is to add coverage for details that are easy to miss.

### Context
{text}

### Coverage Task
Generate up to {n_questions} additional training examples that emphasize:
- named abilities, items, stats, modifiers, or keywords
- numeric thresholds, prerequisites, exceptions, or limits
- definitions of terms or subsystems

Avoid generic summaries. Focus on precise, testable Q/A that can be answered from the text.
Use the task type: {task_type}
Allowed task types: {task_types}

### Output Format
[
  {{
    "instruction": "...",
    "output": "...",
    "task_type": "rules_qa"
  }}
]

Return JSON list only. Do not include markdown fences.
"""


def generate_coverage_pairs(
    text_chunk: str,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    task_type: str,
    allowed_task_types: List[str],
    n_questions: int,
    repair_invalid_json: bool,
    invalid_log_path: Optional[str],
    warning_limiter: Optional[WarningLimiter],
) -> List[Dict[str, str]]:
    if n_questions <= 0:
        return []

    prompt = COVERAGE_PROMPT_TEMPLATE.format(
        text=text_chunk,
        n_questions=n_questions,
        task_type=task_type,
        task_types=", ".join(allowed_task_types),
    )

    response = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    if not response.strip():
        msg = "Warning: Empty coverage response from model."
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return []

    parsed = parse_json_list(response, warning_limiter=warning_limiter)
    if parsed is not None:
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
    return repaired_parsed
