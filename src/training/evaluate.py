import torch
from unsloth import FastLanguageModel
import argparse
import yaml

def evaluate(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    max_seq_length = config['model'].get('max_seq_length', 2048)
    dtype = None
    load_in_4bit = True

    model_path = config['training']['output_dir']
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # Load the fine-tuned adapter
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

    # Test questions
    questions = [
        "How does Overcharge work in Lancer?",
        "What are the traits of the Horus manufacturer?",
        "Explain the rules for hiding during combat."
    ]

    for q in questions:
        print("-" * 30)
        print(f"Instruction: {q}")
        inputs = tokenizer(
            [alpaca_prompt.format(q, "", "")], return_tensors = "pt"
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
        response = tokenizer.batch_decode(outputs)
        print(f"Response:\n{response[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rpg_finetune.yaml", help="Path to config file")
    args = parser.parse_args()
    
    evaluate(args.config)
