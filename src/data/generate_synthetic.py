import json
import argparse
import os
import random
from typing import List, Dict
import yaml
from tqdm import tqdm
from ..utils.llm_client import call_llm

# Advanced prompt with Chain-of-Thought (CoT) and explicit reasoning steps
PROMPT_TEMPLATE = """
You are an expert Game Master and Rules Lawyer for the Lancer RPG system.
Your goal is to create high-quality, logically consistent training data for a new AI model.

### Context
Read the following text from the Lancer Core Book:
{text}

### Task
Generate {n_questions} training examples based on the text above. 
Each example must be a pair of "instruction" (a user question or prompt) and "output" (the ideal response).

### Requirements
1. **Variety**: Create a mix of:
   - **Rule Clarifications**: "How does X interact with Y?"
   - **Tactical Scenarios**: "I'm in situation Z, what can I do?"
   - **Lore/Flavor**: "Describe the history of..."
2. **Reasoning**: Before writing the final JSON, you must THINK step-by-step to ensure the answer is correct according to the rules provided.
3. **Format**: Output a valid JSON list of objects. Each object must have:
   - `instruction`: The user prompt.
   - `output`: The correct, high-quality answer.
   - `thought_process`: (Optional but recommended) Your internal reasoning verification.

### Output Format
[
  {{
    "instruction": "...",
    "output": "...",
    "thought_process": "Checked page X, rule says Y..."
  }}
]

Do not include any markdown formatting (like ```json) outside the standard response if possible, just the raw JSON list.
"""

def generate_qa_pairs(text_chunk: str, n_questions: int = 2) -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(text=text_chunk, n_questions=n_questions)
    
    # Call the model (GPT-5.1-Thinking or similar)
    response = call_llm(prompt, model="gpt-5.1-thinking")
    
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
        
    # Determine input path from config or default
    ingest_config = config.get("ingest", {})
    input_path = ingest_config.get("raw_output_path", "dataset/raw_extracted.json")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Did you run ingest_pdf.py?")
        return
        
    with open(input_path, "r") as f:
        chunks = json.load(f)
        
    output_data = []
    
    # Configurable limits
    max_samples = config.get("n_samples", 50)
    print(f"Generating up to {max_samples} synthetic samples from {len(chunks)} chunks...")
    
    # Shuffle chunks to get random distribution of rules if we hit the limit
    random.shuffle(chunks)
    
    count = 0
    pbar = tqdm(total=max_samples)
    
    for chunk in chunks:
        if count >= max_samples:
            break
            
        # Skip empty or very short chunks (e.g., page numbers or headers)
        if len(chunk["text"]) < 200:
            continue
            
        # Generate data
        qa_pairs = generate_qa_pairs(chunk["text"], n_questions=2)
        
        for pair in qa_pairs:
            if count >= max_samples:
                break
                
            record = {
                "instruction": pair["instruction"],
                "input": "", 
                "output": pair["output"],
                "source_page": chunk["page"],
                "generator_thought": pair.get("thought_process", "")
            }
            output_data.append(record)
            count += 1
            pbar.update(1)
            
    pbar.close()
            
    output_path = config.get("output", {}).get("path", "dataset/lancer_synthetic.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(output_data)} pairs. Saved to {output_path}")

if __name__ == "__main__":
    main()
