#!/usr/bin/env python3
"""
Run the Lancer evaluation benchmark against a trained model.

Usage:
    # With local Ollama model
    python scripts/run_eval_benchmark.py --model ollama:lancer-expert
    
    # With HuggingFace model
    python scripts/run_eval_benchmark.py --model hf:your-username/lancer-rules-7b-lora
    
    # Dry run (print questions without inference)
    python scripts/run_eval_benchmark.py --dry-run
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.evaluate_rpg import (
    load_eval_set,
    run_benchmark,
)


def create_ollama_inference_fn(model_name: str):
    """Create inference function using Ollama."""
    try:
        import ollama
    except ImportError:
        print("❌ Error: ollama package not installed. Run: pip install ollama")
        sys.exit(1)
    
    def inference_fn(question: str, context: str) -> str:
        prompt = f"""You are a grounded RPG assistant. Use only the provided context to answer.
If the context does not contain the answer, say "Not found in context."
Include citations using page numbers.

Context:
{context}

Question: {question}

Answer:"""
        
        response = ollama.generate(model=model_name, prompt=prompt)
        return response["response"]
    
    return inference_fn


def create_hf_inference_fn(model_path: str):
    """Create inference function using HuggingFace/Unsloth model."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ Error: unsloth not installed. This requires GPU environment.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    def inference_fn(question: str, context: str) -> str:
        user_msg = f"{question}\n\nContext:\n{context}"
        conversation = [
            {"role": "system", "content": "You are a grounded RPG assistant. Use only the provided context to answer. If the context does not contain the answer, say 'Not found in context.' Include citations."},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0]
        
        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        return response.strip()
    
    return inference_fn


def dry_run(eval_path: str):
    """Print evaluation examples without running inference."""
    examples = load_eval_set(eval_path)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION BENCHMARK: {len(examples)} examples")
    print(f"{'='*60}\n")
    
    by_task = {}
    by_difficulty = {}
    answerable_count = 0
    
    for ex in examples:
        by_task[ex.task_type] = by_task.get(ex.task_type, 0) + 1
        by_difficulty[ex.difficulty] = by_difficulty.get(ex.difficulty, 0) + 1
        if ex.answerable:
            answerable_count += 1
    
    print("By Task Type:")
    for task, count in sorted(by_task.items()):
        print(f"  {task}: {count}")
    
    print("\nBy Difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        print(f"  {diff}: {count}")
    
    print(f"\nAnswerable: {answerable_count}")
    print(f"Unanswerable (refusal tests): {len(examples) - answerable_count}")
    
    print(f"\n{'='*60}")
    print("SAMPLE QUESTIONS:")
    print(f"{'='*60}\n")
    
    for i, ex in enumerate(examples[:5]):
        print(f"[{ex.id}] ({ex.difficulty}, {ex.task_type})")
        print(f"Q: {ex.question}")
        print(f"Key facts: {ex.key_facts}")
        print(f"Answerable: {ex.answerable}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run Lancer evaluation benchmark")
    parser.add_argument(
        "--eval-set",
        default="dataset/eval_benchmark.yaml",
        help="Path to evaluation set",
    )
    parser.add_argument(
        "--model",
        help="Model to evaluate. Format: 'ollama:model-name' or 'hf:model-path'",
    )
    parser.add_argument(
        "--output",
        default="dataset/eval_results.md",
        help="Path to save results markdown",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print evaluation set info without running inference",
    )
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run(args.eval_set)
        return
    
    if not args.model:
        print("❌ Error: --model required (e.g., 'ollama:lancer-expert' or 'hf:user/model')")
        sys.exit(1)
    
    # Parse model specification
    if args.model.startswith("ollama:"):
        model_name = args.model.replace("ollama:", "")
        print(f"Using Ollama model: {model_name}")
        inference_fn = create_ollama_inference_fn(model_name)
    elif args.model.startswith("hf:"):
        model_path = args.model.replace("hf:", "")
        print(f"Using HuggingFace model: {model_path}")
        inference_fn = create_hf_inference_fn(model_path)
    else:
        print("❌ Error: Unknown model format. Use 'ollama:name' or 'hf:path'")
        sys.exit(1)
    
    # Run benchmark
    results = run_benchmark(
        eval_set_path=args.eval_set,
        inference_fn=inference_fn,
        output_path=args.output,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total Examples: {results.total_examples}")
    print(f"Accuracy:       {results.avg_accuracy:.1%}")
    print(f"Grounding:      {results.avg_grounding:.1%}")
    print(f"Citation:       {results.avg_citation:.1%}")
    print(f"Refusal:        {results.avg_refusal:.1%}")
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

