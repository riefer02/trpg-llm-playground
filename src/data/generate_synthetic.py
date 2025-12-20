import json
import argparse
import os
import random
import re
import itertools
from datetime import datetime, timezone
from typing import Optional
from typing import List, Dict
import yaml
from tqdm import tqdm
from ..utils.llm_client import call_llm

LOW_SIGNAL_KEYWORDS = [
    "all rights reserved",
    "isbn",
    "copyright",
    "credits",
    "printed in",
    "graphic design",
    "art direction",
    "layout",
    "www.",
    "http",
]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def is_table_like(lines: List[str]) -> bool:
    if len(lines) < 5:
        return False
    table_like_lines = sum(
        1 for line in lines if re.search(r"\s{2,}|\t|\|", line)
    )
    ratio = table_like_lines / max(1, len(lines))
    return ratio >= 0.3


def analyze_text(text: str) -> Dict[str, object]:
    text_len = len(text)
    lines = [line for line in text.splitlines() if line.strip()]
    alpha = sum(c.isalpha() for c in text)

    alpha_ratio = alpha / max(1, text_len)
    lower_text = text.lower()
    has_low_signal_keyword = any(k in lower_text for k in LOW_SIGNAL_KEYWORDS)
    table_like = is_table_like(lines)

    low_signal = (
        text_len < 200
        or (alpha_ratio < 0.5 and text_len < 1200)
        or (has_low_signal_keyword and text_len < 1200)
    )

    return {
        "text_len": text_len,
        "lines": lines,
        "table_like": table_like,
        "low_signal": low_signal,
    }


def suggest_questions(text_len: int, table_like: bool, low_signal: bool) -> int:
    if low_signal:
        return 0
    base = clamp(text_len // 800 + 2, 2, 6)
    if table_like and base < 6:
        base += 1
    return base


# Advanced prompt with Chain-of-Thought (CoT) and explicit reasoning steps
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
) -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(
        text=text_chunk,
        n_questions=n_questions,
        task_type=task_type,
        task_types=", ".join(allowed_task_types),
        extra_instructions=extra_instructions,
    )
    
    # Call the model (GPT-5.1-Thinking or similar)
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
    
    # Robust parsing logic
    try:
        # 1. Clean markdown code blocks if present
        clean_response = response.replace("```json", "").replace("```", "").strip()
        
        # 2. Find list start/end
        start_idx = clean_response.find('[')
        end_idx = clean_response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = clean_response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print(f"Warning: Could not find JSON list in response. First 50 chars: {clean_response[:50]}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Response snippet: {response[:100]}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q/A pairs from extracted text.")
    parser.add_argument("--config", type=str, default="config/synthetic_generic.yaml", help="Path to config YAML.")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    debug_config = config.get("debug", {}) or {}
        
    output_config = config.get("output", {}) or {}
    run_id = output_config.get("run_id", "auto")
    if run_id == "auto":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    path_vars = {
        "project_name": config.get("project_name", "default"),
        "dataset_tag": config.get("dataset_tag", "v1"),
        "run_id": run_id,
    }
        
    # Determine input path from config or default
    ingest_config = config.get("ingest", {})
    raw_path_template = ingest_config.get("raw_output_path", "dataset/raw_extracted.json")
    input_path = raw_path_template.format(**path_vars)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Did you run ingest_pdf.py?")
        return
        
    with open(input_path, "r") as f:
        chunks = json.load(f)
        
    output_data = []
    
    # Configurable limits
    max_samples = config.get("n_samples", 50)
    limits_config = config.get("limits", {}) or {}
    enforce_max_samples = limits_config.get("enforce_max_samples", True)
    if debug_config.get("enabled"):
        debug_max = debug_config.get("max_samples")
        if isinstance(debug_max, int) and debug_max > 0:
            max_samples = min(max_samples, debug_max)
            print(f"Debug mode: limiting synthetic samples to {max_samples}.")

    if enforce_max_samples:
        print(f"Generating up to {max_samples} synthetic samples from {len(chunks)} chunks...")
    else:
        max_samples = None
        print(f"Generating synthetic samples from {len(chunks)} chunks...")
    
    # Shuffle chunks to get random distribution of rules if we hit the limit
    random.shuffle(chunks)
    
    count = 0
    pbar = tqdm(total=max_samples)
    
    llm_config = config.get("llm", {}) or {}
    model = llm_config.get("model", "gpt-5-mini")
    temperature = llm_config.get("temperature")
    max_output_tokens = llm_config.get("max_output_tokens")
    max_completion_tokens = llm_config.get("max_completion_tokens")

    task_types = config.get("task_types") or DEFAULT_TASK_TYPES
    task_types = [t for t in task_types if isinstance(t, str) and t.strip()]
    if not task_types:
        task_types = DEFAULT_TASK_TYPES
    random.shuffle(task_types)
    task_type_cycle = itertools.cycle(task_types)

    skipped_low_signal = 0
    table_like_pages = 0
    processed_pages = 0
    requested_questions = 0

    for chunk in chunks:
        if max_samples is not None and count >= max_samples:
            break
            
        analysis = analyze_text(chunk["text"])
        if analysis["low_signal"]:
            skipped_low_signal += 1
            continue

        n_questions = suggest_questions(
            analysis["text_len"],
            analysis["table_like"],
            analysis["low_signal"],
        )
        if n_questions <= 0:
            skipped_low_signal += 1
            continue

        if max_samples is not None:
            remaining = max_samples - count
            n_questions = min(n_questions, remaining)
        requested_questions += n_questions

        extra_instructions = ""
        if analysis["table_like"]:
            table_like_pages += 1
            extra_instructions = (
                "### Table Hint\n"
                "If the text looks tabular, extract key rows into compact Q/A pairs."
            )

        processed_pages += 1

        # Generate data
        task_type = next(task_type_cycle)
        qa_pairs = generate_qa_pairs(
            chunk["text"],
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            task_type=task_type,
            allowed_task_types=task_types,
            extra_instructions=extra_instructions,
            n_questions=n_questions,
        )
        
        for pair in qa_pairs:
            if max_samples is not None and count >= max_samples:
                break

            missing = []
            if "instruction" not in pair:
                missing.append("instruction")
            if "output" not in pair:
                missing.append("output")
            if missing:
                page = chunk.get("page", "unknown")
                print(f"Warning: Skipping malformed item on page {page}; missing {missing}.")
                continue

            pair_task_type = pair.get("task_type", task_type)
            if pair_task_type not in task_types:
                page = chunk.get("page", "unknown")
                print(
                    f"Warning: Invalid task_type '{pair_task_type}' on page {page}; "
                    f"forcing to '{task_type}'."
                )
                pair_task_type = task_type

            record = {
                "instruction": pair["instruction"],
                "input": "", 
                "output": pair["output"],
                "task_type": pair_task_type,
                "source_page": chunk["page"]
            }
            output_data.append(record)
            count += 1
            pbar.update(1)
            
    pbar.close()
            
    output_path_template = output_config.get("path", "dataset/{project_name}_{dataset_tag}_synthetic.jsonl")
    output_path = output_path_template.format(**path_vars)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    append_output = output_config.get("append", False)
    if os.path.exists(output_path) and not append_output:
        print(f"Warning: Output file already exists and will be overwritten: {output_path}")
        print("Tip: Set output.append: true or include {run_id} in output.path to avoid overwrites.")
    
    # Validation: Check approx token lengths
    print("\n--- Data Stats ---")
    long_samples = 0
    total_samples = len(output_data)
    # Rough estimate: 1 token ~= 4 chars
    limit_4k = 4096 * 4 
    
    for record in output_data:
        # Input context is the biggest factor
        combined_len = len(record.get("input", "")) + len(record.get("instruction", "")) + len(record.get("output", ""))
        if combined_len > limit_4k:
            long_samples += 1
            
    print(f"Total Samples: {total_samples}")
    print(f"Samples > ~4096 tokens (est): {long_samples}")
    print(f"Low-signal pages skipped: {skipped_low_signal}")
    print(f"Table-like pages processed: {table_like_pages}")
    if processed_pages:
        avg_requested = requested_questions / processed_pages
        print(f"Avg requested questions per processed page: {avg_requested:.2f}")
    if long_samples > 0:
        print("⚠️ Warning: Some samples might be truncated on small context models (e.g. Llama-3-8B).")
    else:
        print("✅ All samples fit comfortably within standard 4k context.")

    write_mode = "a" if append_output else "w"
    with open(output_path, write_mode, encoding="utf-8") as f:
        for record in output_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(output_data)} pairs. Saved to {output_path}")

if __name__ == "__main__":
    main()
