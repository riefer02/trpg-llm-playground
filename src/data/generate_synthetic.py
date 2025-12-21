import argparse
import itertools
import json
import os
import random
from datetime import datetime, timezone

import yaml
from tqdm import tqdm

from .synth_analysis import analyze_text, suggest_questions
from .synth_io import load_chunks
from .synth_llm import DEFAULT_TASK_TYPES, WarningLimiter, generate_qa_pairs
from .synth_resume import build_signature, load_checkpoint, save_checkpoint


def main() -> None:
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

    chunks = load_chunks(input_path)
    if not isinstance(chunks, list):
        print(f"Error: Input file {input_path} did not contain a list of chunks.")
        return

    generation_config = config.get("generation", {}) or {}
    shuffle_enabled = generation_config.get("shuffle", True)
    shuffle_seed = generation_config.get("shuffle_seed", 1337)

    order = list(range(len(chunks)))
    if shuffle_enabled:
        rng = random.Random(shuffle_seed)
        rng.shuffle(order)

    task_types = config.get("task_types") or DEFAULT_TASK_TYPES
    task_types = [t for t in task_types if isinstance(t, str) and t.strip()]
    if not task_types:
        task_types = DEFAULT_TASK_TYPES

    resume_config = config.get("resume", {}) or {}
    resume_enabled = resume_config.get("enabled", False)
    checkpoint_path = resume_config.get("checkpoint_path", "")
    checkpoint_path = checkpoint_path.format(**path_vars) if checkpoint_path else ""
    resume_allow_mismatch = resume_config.get("allow_mismatch", False)
    resume_force_restart = resume_config.get("force_restart", False)
    start_index = 0
    checkpoint = None

    if resume_enabled and checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        if resume_force_restart:
            print("Resume force_restart enabled; starting fresh.")
            checkpoint = None
        elif checkpoint and checkpoint.get("run_id"):
            run_id = checkpoint["run_id"]
            path_vars["run_id"] = run_id

    output_path_template = output_config.get(
        "path",
        "dataset/{project_name}_{dataset_tag}_synthetic.jsonl",
    )
    output_path = output_path_template.format(**path_vars)

    signature = build_signature(
        config,
        input_path,
        output_path,
        task_types,
        shuffle_enabled,
        shuffle_seed,
    )

    if resume_enabled and checkpoint_path and checkpoint and not resume_force_restart:
        if checkpoint.get("signature") == signature:
            start_index = int(checkpoint.get("next_index", 0))
        elif resume_allow_mismatch:
            print("Warning: Checkpoint signature mismatch; resuming anyway (allow_mismatch=true).")
            start_index = int(checkpoint.get("next_index", 0))
        else:
            print("Warning: Checkpoint signature mismatch; starting fresh.")
            checkpoint = None
            start_index = 0
            if output_config.get("run_id", "auto") == "auto":
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                path_vars["run_id"] = run_id
                output_path = output_path_template.format(**path_vars)
                signature = build_signature(
                    config,
                    input_path,
                    output_path,
                    task_types,
                    shuffle_enabled,
                    shuffle_seed,
                )

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

    count = 0
    long_samples = 0
    total_samples = 0
    if checkpoint and checkpoint.get("stats"):
        stats = checkpoint.get("stats", {})
        count = int(stats.get("samples_written", 0))
        total_samples = int(stats.get("samples_written", 0))
        long_samples = int(stats.get("long_samples", 0))
    elif checkpoint and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f if _.strip())
            total_samples = count
        except Exception as e:
            print(f"Warning: Could not count existing output lines: {e}")

    pbar = tqdm(total=max_samples, initial=count)

    llm_config = config.get("llm", {}) or {}
    model = llm_config.get("model", "gpt-5-mini")
    temperature = llm_config.get("temperature")
    max_output_tokens = llm_config.get("max_output_tokens")
    max_completion_tokens = llm_config.get("max_completion_tokens")
    repair_invalid_json = llm_config.get("repair_invalid_json", True)
    invalid_response_log = llm_config.get("invalid_response_log")

    logging_config = config.get("logging", {}) or {}
    max_warnings = logging_config.get("max_warnings", 10)
    hud_every = logging_config.get("hud_every", 10)
    quiet = logging_config.get("quiet", False)
    warning_limiter = WarningLimiter(0 if quiet else max_warnings)

    random.shuffle(task_types)
    task_type_cycle = itertools.cycle(task_types)
    if start_index > 0:
        for _ in range(start_index):
            next(task_type_cycle)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    append_output = output_config.get("append", False)
    flush_every = output_config.get("flush_every", 50)
    if resume_enabled and checkpoint and start_index > 0:
        append_output = True
    if os.path.exists(output_path) and not append_output:
        print(f"Warning: Output file already exists and will be overwritten: {output_path}")
        print("Tip: Set output.append: true or include {run_id} in output.path to avoid overwrites.")

    write_mode = "a" if append_output else "w"
    checkpoint_stats = checkpoint.get("stats", {}) if checkpoint else {}
    skipped_low_signal = int(checkpoint_stats.get("skipped_low_signal", 0))
    table_like_pages = int(checkpoint_stats.get("table_like_pages", 0))
    processed_pages = int(checkpoint_stats.get("processed_pages", 0))
    requested_questions = int(checkpoint_stats.get("requested_questions", 0))

    with open(output_path, write_mode, encoding="utf-8") as f:
        for order_pos in range(start_index, len(order)):
            if max_samples is not None and count >= max_samples:
                break

            chunk = chunks[order[order_pos]]

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
                repair_invalid_json=repair_invalid_json,
                invalid_log_path=invalid_response_log,
                warning_limiter=warning_limiter,
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
                    "source_page": chunk["page"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                total_samples += 1
                combined_len = (
                    len(record.get("input", ""))
                    + len(record.get("instruction", ""))
                    + len(record.get("output", ""))
                )
                if combined_len > 4096 * 4:
                    long_samples += 1
                if flush_every and count % flush_every == 0:
                    f.flush()
                pbar.update(1)

            if resume_enabled and checkpoint_path:
                checkpoint_payload = {
                    "run_id": run_id,
                    "signature": signature,
                    "next_index": order_pos + 1,
                    "stats": {
                        "samples_written": total_samples,
                        "long_samples": long_samples,
                        "skipped_low_signal": skipped_low_signal,
                        "table_like_pages": table_like_pages,
                        "processed_pages": processed_pages,
                        "requested_questions": requested_questions,
                    },
                }
                save_checkpoint(checkpoint_path, checkpoint_payload)

            if hud_every and processed_pages % hud_every == 0:
                avg_requested = requested_questions / max(1, processed_pages)
                pbar.set_postfix(
                    {
                        "pages": processed_pages,
                        "written": total_samples,
                        "skipped": skipped_low_signal,
                        "tables": table_like_pages,
                        "avg_q": f"{avg_requested:.2f}",
                    }
                )

    pbar.close()

    # Validation: Check approx token lengths
    print("\n--- Data Stats ---")
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

    print(f"Successfully generated {total_samples} pairs. Saved to {output_path}")


if __name__ == "__main__":
    main()
