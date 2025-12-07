import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import yaml
import argparse
import os

def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Model
    max_seq_length = config['model'].get('max_seq_length', 2048)
    dtype = None # Auto detection
    load_in_4bit = config['model'].get('load_in_4bit', True)

    print(f"Loading model: {config['model']['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model']['base_model'],
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
    dataset_path = config['dataset']['train_path']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    # Formatting function
    # Check if the model is a Chat model (like Qwen/Llama Instruct) to use the right template
    # Unsloth handles some of this, but explicit templates are safer.
    # For Qwen, standard ChatML or the Alpaca format often works, but let's stick to Alpaca for simplicity
    # or detect if we need a specific chat template.
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir = config['training']['output_dir'],
        per_device_train_batch_size = config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
        warmup_steps = config['training']['warmup_steps'],
        max_steps = config['training']['max_steps'],
        learning_rate = float(config['training']['learning_rate']),
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = config['training']['logging_steps'],
        save_steps = config['training'].get('save_steps', 0), # Support save_steps from config
        optim = config['training']['optim'],
        weight_decay = config['training']['weight_decay'],
        lr_scheduler_type = config['training']['lr_scheduler_type'],
        seed = config['training']['seed'],
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can speed up training for short sequences
        args = training_args,
    )

    # 5. Train
    print("Starting training...")
    trainer_stats = trainer.train()

    # 6. Save
    print(f"Saving model to {config['training']['output_dir']}")
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rpg_finetune.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
