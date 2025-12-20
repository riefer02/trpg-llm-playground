import json
from typing import Dict, List, Optional

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response

DEFAULT_TASK_TYPES = [
    "rules_qa",
    "character_build",
    "scenario_seed",
    "gm_guidance",
    "lore",
]

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
3. **Format**: Output a valid JSON list of objects. Each object must have:
   - `instruction`: The user prompt.
   - `output`: The correct, high-quality answer.
   - `task_type`: One of: {task_types}

### Output Format
[
  {{
    "instruction": "...",
    "output": "...",
    "task_type": "rules_qa"
  }}
]

Do not include any markdown formatting (like ```json) outside the standard response if possible, just the raw JSON list.
"""


def parse_json_list(response: str) -> Optional[List[Dict[str, str]]]:
    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        start_idx = clean_response.find("[")
        end_idx = clean_response.rfind("]") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = clean_response[start_idx:end_idx]
            return json.loads(json_str)
        print(f"Warning: Could not find JSON list in response. First 50 chars: {clean_response[:50]}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Response snippet: {response[:100]}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def repair_json_response(
    response: str,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
) -> str:
    repair_prompt = (
        "Fix the following output so it is a valid JSON list of objects. "
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
    print("Warning: JSON repair returned empty response.")
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
) -> List[Dict[str, str]]:
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
    )
    if not response.strip():
        print("Warning: Empty response from model. Skipping this chunk.")
        return []

    parsed = parse_json_list(response)
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
    )
    if not repaired:
        return []

    repaired_parsed = parse_json_list(repaired)
    if repaired_parsed is None:
        if invalid_log_path:
            log_invalid_response(invalid_log_path, repaired)
        return []
    return repaired_parsed
