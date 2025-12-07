import json
import argparse
import os
import random
from typing import List, Dict
import yaml
from tqdm import tqdm
from ..utils.llm_client import call_llm

PROMPT_TEMPLATE = """
You are an expert on the Lancer RPG system. 
Read the following text from the Lancer Core Book:

{text}

Based on this text, generate {n_questions} question-answer pairs that would help a new player understand the rules or lore.
Format the output as a JSON list of objects with "instruction" (the question) and "output" (the answer).
Do not include any other text.
"""

def generate_qa_pairs(text_chunk: str, n_questions: int = 2) -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(text=text_chunk, n_questions=n_questions)
    response = call_llm(prompt)
    
    # robust parsing of the LLM response
    try:
        # In a real scenario, we'd need better parsing or structured output enforcement
        # For this stub, we'll return mock data if the response is the mock string
        if response.startswith("Mock response"):
            return [
                {"instruction": f"Mock Question regarding: {text_chunk[:20]}...", "output": "Mock Answer..."}
                for _ in range(n_questions)
            ]
        
        # Try to parse actual JSON response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print("Could not find JSON in response")
            return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q/A pairs from extracted text.")
    parser.add_argument("--input_path", type=str, default="dataset/raw_extracted.json", help="Path to input extracted JSON.")
    parser.add_argument("--config", type=str, default="config/synthetic_generic.yaml", help="Path to config YAML.")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    with open(args.input_path, "r") as f:
        chunks = json.load(f)
        
    output_data = []
    
    # Limit for testing if needed, or process all
    print(f"Generating synthetic data from {len(chunks)} chunks...")
    
    for chunk in tqdm(chunks):
        # Skip empty or very short chunks
        if len(chunk["text"]) < 100:
            continue
            
        qa_pairs = generate_qa_pairs(chunk["text"], n_questions=2)
        
        for pair in qa_pairs:
            # Add system prompt or input field if needed for the instruction format
            record = {
                "instruction": pair["instruction"],
                "input": "", # Context could go here if we wanted RAG-style, but for now empty
                "output": pair["output"],
                "source_page": chunk["page"]
            }
            output_data.append(record)
            
    output_path = config.get("output", {}).get("path", "dataset/lancer_synthetic.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(output_data)} pairs. Saved to {output_path}")

if __name__ == "__main__":
    main()

