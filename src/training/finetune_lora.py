import sys
import os
import argparse
import yaml
import torch

# Graceful import check for Unsloth
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("❌ Error: Unsloth is not installed.")
    print("Please install it: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    sys.exit(1)

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def train(config_path: str):
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_vars = {
        "project_name": config.get("project_name", "default"),
        "dataset_tag": config.get("dataset_tag", "v1")
    }
    print(f"Initializing training for project: {path_vars['project_name']} ({path_vars['dataset_tag']})")

    # 1. Load Model
    max_seq_length = config['model'].get('max_seq_length', 2048)
    dtype = None # Auto detection
    load_in_4bit = config['model'].get('load_in_4bit', True)
    model_name = config['model']['base_model']

    print(f"Loading model: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = config['lora']['lora_alpha'],
        lora_dropout = config['lora']['lora_dropout'], 
        bias = config['lora']['bias'],
        use_gradient_checkpointing = config['lora']['use_gradient_checkpointing'],
        random_state = config['lora']['random_state'],
        use_rslora = config['lora']['use_rslora'],
        loftq_config = config['lora']['loftq_config'],
    )

    # 3. Load Dataset
    dataset_path_template = config['dataset']['train_path']
    dataset_path = dataset_path_template.format(**path_vars)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found at: {dataset_path}")
        print("Did you run the synthetic data generation step (Cell 5) successfully?")
        sys.exit(1)
        
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    # 4. Apply Chat Template (Best Practice for Qwen/Llama)
    # Unsloth/Transformers can auto-detect the right template for the model
    # This maps "instruction/input/output" columns to the standard chat format
    
    def formatting_prompts_func(examples):
        convos = []
        texts = []
        mapper = {"system": "You are a helpful assistant.", "user": "", "assistant": ""}
        
        for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # Construct the conversation
            # If 'input' exists, append it to instruction
            user_msg = instruction
            if input and str(input).strip():
                user_msg += f"\n\nContext:\n{input}"
                
            conversation = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
            
            # Apply the model's specific chat template (ChatML for Qwen, etc.)
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 5. Training Arguments
    output_dir_template = config['training']['output_dir']
    output_dir = output_dir_template.format(**path_vars)
    
    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
        warmup_steps = config['training']['warmup_steps'],
        max_steps = config['training']['max_steps'],
        learning_rate = float(config['training']['learning_rate']),
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = config['training']['logging_steps'],
        save_steps = config['training'].get('save_steps', 0),
        optim = config['training']['optim'],
        weight_decay = config['training']['weight_decay'],
        lr_scheduler_type = config['training']['lr_scheduler_type'],
        seed = config['training']['seed'],
        report_to = config['training'].get('report_to', "none"),
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = training_args,
    )

    # 6. Train
    print("Starting training...")
    trainer_stats = trainer.train()

    # 7. Save Artifacts (Model + Recipe)
    print(f"Saving model and artifacts to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # PRO TIP: Save the exact config used to generate this model
    # This ensures you always know the "recipe" for this specific artifact
    with open(os.path.join(output_dir, "training_config_captured.yaml"), "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rpg_finetune.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
