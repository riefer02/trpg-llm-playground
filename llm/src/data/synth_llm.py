import json
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response
from .synth_prompts import PromptConfig

DEFAULT_TASK_TYPES = [
    "rules_qa",
    "character_build",
    "scenario_seed",
    "gm_guidance",
    "lore",
]


class SyntheticExample(BaseModel):
    """A single synthetic Q/A example."""

    instruction: str = Field(description="The user prompt or question.")
    output: str = Field(description="The ideal, high-quality response.")
    task_type: str = Field(description="The category of the task (e.g., rules_qa, lore).")


class SyntheticExampleList(BaseModel):
    """A list of synthetic Q/A examples."""

    examples: List[SyntheticExample]


# Legacy template kept for backwards compatibility - prefer PromptConfig
PROMPT_TEMPLATE = """
You are an expert Game Master and Rules Lawyer for the Lancer RPG system.
Your goal is to create high-quality, logically consistent training data for a new AI model.

### Context
Read the following text from the Lancer Core Book:
{text}

### Task
Generate {n_questions} training examples based on the text above. 
Each example must be a pair of "instruction" (a user question or prompt) and "output" (the ideal response).

### Task Type
The task type for this generation is: {task_type}
Choose prompts and answers that match this task type.

{extra_instructions}

### Requirements
1. **Variety**: Create a mix of:
   - **Rule Clarifications**: "How does X interact with Y?"
   - **Tactical Scenarios**: "I'm in situation Z, what can I do?"
   - **Lore/Flavor**: "Describe the history of..."
2. **Reasoning**: Think step-by-step internally, but do not include reasoning in the output.

### Format
Output MUST be a valid JSON object with an "examples" key containing a list of objects. Each object must have:
- `instruction`: The user prompt.
- `output`: The correct, high-quality answer.
- `task_type`: One of: {task_types}

### Output Format
{{
  "examples": [
    {{
      "instruction": "...",
      "output": "...",
      "task_type": "rules_qa"
    }}
  ]
}}

Do not include any markdown formatting (like ```json) outside the response. Return JSON only.
"""


class WarningLimiter:
    def __init__(self, max_warnings: int) -> None:
        self.max_warnings = max_warnings
        self.count = 0
        self.suppressed = False

    def warn(self, message: str) -> None:
        if self.max_warnings <= 0:
            return
        if self.count < self.max_warnings:
            print(message)
            self.count += 1
            if self.count == self.max_warnings and not self.suppressed:
                print("Warning: Further warnings suppressed.")
                self.suppressed = True


def _inject_task_type_if_missing(item: dict, default_task_type: Optional[str]) -> dict:
    """Inject task_type into an item if missing and default is provided."""
    if "task_type" not in item and default_task_type:
        item = dict(item)
        item["task_type"] = default_task_type
    return item


def parse_json_list(
    response: str,
    warning_limiter: Optional[WarningLimiter] = None,
    default_task_type: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        # Look for the start of a JSON object or list
        start_obj = clean_response.find("{")
        start_list = clean_response.find("[")

        if start_obj != -1 and (start_list == -1 or start_obj < start_list):
            # Probably an object like {"examples": [...]}
            end_idx = clean_response.rfind("}") + 1
            json_str = clean_response[start_obj:end_idx]
            data = json.loads(json_str)
            if isinstance(data, dict) and "examples" in data:
                # Inject missing task_type before validation
                examples = [_inject_task_type_if_missing(ex, default_task_type) for ex in data["examples"]]
                data["examples"] = examples
                # Validate with Pydantic
                validated = SyntheticExampleList(**data)
                return [item.model_dump() for item in validated.examples]
        elif start_list != -1:
            # Fallback for old list format
            end_idx = clean_response.rfind("]") + 1
            json_str = clean_response[start_list:end_idx]
            data = json.loads(json_str)
            if isinstance(data, list):
                # Inject missing task_type before validation
                data = [_inject_task_type_if_missing(item, default_task_type) for item in data]
                # Validate each item
                validated_list = [SyntheticExample(**item).model_dump() for item in data]
                return validated_list

        msg = f"Warning: Could not find valid JSON examples in response. Snippet: {clean_response[:100]}"
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return None
    except Exception as e:
        msg = f"Error parsing or validating JSON: {e}. Snippet: {response[:100]}"
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return None


def repair_json_response(
    response: str,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    warning_limiter: Optional[WarningLimiter] = None,
    task_type: Optional[str] = None,
) -> str:
    # Include task_type requirement in repair prompt for better results
    task_type_hint = f' Each object must have "task_type": "{task_type}".' if task_type else ""
    repair_prompt = (
        "Fix the following output so it is a valid JSON list of objects with "
        f'"instruction", "output", and "task_type" fields.{task_type_hint} '
        "Do not add commentary or extra text. Return JSON only.\n\n"
        f"Output to fix:\n{response}"
    )
    repaired = call_llm(
        repair_prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    if repaired.strip():
        return repaired
    msg = "Warning: JSON repair returned empty response."
    if warning_limiter:
        warning_limiter.warn(msg)
    else:
        print(msg)
    return ""


def generate_qa_pairs(
    text_chunk: str,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    task_type: str,
    allowed_task_types: List[str],
    extra_instructions: str,
    n_questions: int = 2,
    repair_invalid_json: bool = True,
    invalid_log_path: Optional[str] = None,
    warning_limiter: Optional[WarningLimiter] = None,
    prompt_config: Optional[PromptConfig] = None,
) -> List[Dict[str, str]]:
    # Use PromptConfig if provided, otherwise fall back to legacy template
    if prompt_config is not None:
        prompt = prompt_config.format_qa_prompt(
            text=text_chunk,
            n_questions=n_questions,
            task_type=task_type,
            task_types=allowed_task_types,
            extra_instructions=extra_instructions,
        )
    else:
        prompt = PROMPT_TEMPLATE.format(
            text=text_chunk,
            n_questions=n_questions,
            task_type=task_type,
            task_types=", ".join(allowed_task_types),
            extra_instructions=extra_instructions,
        )

    response = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"} if "gpt-4" in model or "gpt-3.5" in model else None,
    )
    if not response.strip():
        print("Warning: Empty response from model. Skipping this chunk.")
        return []

    parsed = parse_json_list(response, warning_limiter=warning_limiter, default_task_type=task_type)
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
        task_type=task_type,
    )
    if not repaired:
        return []

    repaired_parsed = parse_json_list(repaired, warning_limiter=warning_limiter, default_task_type=task_type)
    if repaired_parsed is None:
        if invalid_log_path:
            log_invalid_response(invalid_log_path, repaired)
        return []
    return repaired_parsed
