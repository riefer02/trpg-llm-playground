import argparse
import itertools
import json
import os
import random
from datetime import datetime, timezone

import yaml
from tqdm import tqdm

from .synth_analysis import analyze_text, suggest_questions
from .synth_coverage import generate_coverage_pairs
from .synth_dedup import RunningDeduplicator
from .synth_filter import apply_topic_filter
from .synth_difficulty import (
    DEFAULT_DISTRIBUTION,
    DifficultyStats,
    build_difficulty_prompt_section,
    get_distribution_for_task,
    select_difficulty,
)
from .synth_io import load_chunks
from .synth_llm import DEFAULT_TASK_TYPES, WarningLimiter, generate_qa_pairs
from .synth_multiturn import (
    MultiturnStats,
    generate_multiturn_conversation,
    select_turn_count,
    should_generate_multiturn,
)
from .synth_walkthrough import generate_walkthrough_series
from .synth_negatives import calculate_negative_count, generate_negative_pairs
from .synth_prompts import PromptConfig
from .synth_resume import build_signature, load_checkpoint, save_checkpoint
from .synth_tables import extract_table_pairs
from .synth_verify import VerificationStats, verify_and_filter_pairs

RAG_SYSTEM_PROMPT = (
    "You are a grounded RPG assistant. Use only the provided context to answer. "
    "If the context does not contain the answer, say \"Not found in context.\" and ask "
    "one concise clarifying question. Include citations in a final 'Citations' line "
    "using the provided citation style."
)

RAG_FORMATS = {
    "rules_qa": (
        "### Answer Format\n"
        "Answer: 1-3 sentences.\n"
        "Rules Reference:\n"
        "- Bullet list with citations.\n"
        "Example: short, practical example.\n"
        "Citations: (p. X)"
    ),
    "character_build": (
        "### Answer Format\n"
        "Assumptions: short list.\n"
        "Recommendations: ranked bullets.\n"
        "Tradeoffs: 1-3 bullets.\n"
        "Next Questions: 1-2 clarifying questions.\n"
        "Citations: (p. X) only when referencing mechanics."
    ),
    "scenario_seed": (
        "### Answer Format\n"
        "Scenario Summary: 2-4 sentences.\n"
        "Hooks: 2-3 bullets.\n"
        "Obstacles: 2-3 bullets.\n"
        "Adjustments: easy/hard variants.\n"
        "Citations: (p. X) if mechanics are referenced."
    ),
    "gm_guidance": (
        "### Answer Format\n"
        "Guidance: 3-5 bullets.\n"
        "Table Safety/Clarifications: 1-2 bullets if relevant.\n"
        "Citations: (p. X) for any rule references."
    ),
    "lore": (
        "### Answer Format\n"
        "Answer: 2-4 sentences.\n"
        "Key Facts: 2-4 bullets.\n"
        "Citations: (p. X)"
    ),
}

DEFAULT_RAG_FORMAT = (
    "### Answer Format\n"
    "Answer: concise and grounded.\n"
    "Citations: (p. X)"
)


def build_doc_adjacency(chunks: list) -> tuple[dict, dict]:
    """
    Build prev/next adjacency maps based on document order.

    This ensures prev/next context comes from adjacent text in the source document
    even when we shuffle the sampling order for generation diversity.
    """

    def doc_sort_key(item: dict):
        if isinstance(item.get("page_start"), int):
            return (int(item.get("page_start")), int(item.get("chunk_index", 0)))
        if isinstance(item.get("page"), int):
            return (int(item.get("page")), 0)
        return (10**9, 0)

    sorted_indices = sorted(range(len(chunks)), key=lambda i: doc_sort_key(chunks[i]))
    prev_by_idx = {}
    next_by_idx = {}
    for pos, idx in enumerate(sorted_indices):
        prev_by_idx[idx] = sorted_indices[pos - 1] if pos > 0 else None
        next_by_idx[idx] = sorted_indices[pos + 1] if pos + 1 < len(sorted_indices) else None
    return prev_by_idx, next_by_idx


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

    # Optional: Use rag_ingest chunks instead of raw per-page extraction
    rag_ingest = config.get("rag_ingest", {}) or {}
    chunks_path = None
    if rag_ingest.get("enabled", False):
        chunks_path_tmpl = rag_ingest.get("chunks_output_path")
        if chunks_path_tmpl:
            resolved = chunks_path_tmpl.format(**path_vars)
            if os.path.exists(resolved):
                chunks_path = resolved
                input_path = resolved
                print(f"Using RAG chunks input: {input_path}")
            else:
                print(
                    f"Warning: rag_ingest.enabled is true but chunks file not found at {resolved}. "
                    "Falling back to raw ingest output."
                )

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Did you run ingest_pdf.py?")
        return

    chunks = load_chunks(input_path)
    if not isinstance(chunks, list):
        print(f"Error: Input file {input_path} did not contain a list of chunks.")
        return

    # Apply topic filtering if configured
    topic_filter_config = config.get("topic_filter", {}) or {}
    if topic_filter_config.get("enabled", False):
        original_count = len(chunks)
        chunks = apply_topic_filter(chunks, topic_filter_config)
        print(f"Topic filter: {original_count} -> {len(chunks)} chunks")
        if not chunks:
            print("Error: No chunks remaining after topic filter. Check your filter config.")
            return

    # Build adjacency in document order so prev/next context is stable even when we shuffle sampling order.
    # For page-based raw chunks this is page number; for RAG chunks it is (page_start, chunk_index).
    prev_by_idx, next_by_idx = build_doc_adjacency(chunks)

    generation_config = config.get("generation", {}) or {}
    shuffle_enabled = generation_config.get("shuffle", True)
    shuffle_seed = generation_config.get("shuffle_seed", 1337)
    context_config = config.get("context", {}) or {}

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

    # Initialize PromptConfig with topic from config
    topic = config.get("topic", "the RPG system")
    prompt_config = PromptConfig(config, topic=topic)

    # Quality enhancement configs
    negatives_config = config.get("negatives", {}) or {}
    negatives_enabled = negatives_config.get("enabled", False)
    negatives_ratio = negatives_config.get("ratio", 0.12)
    negatives_max_per_chunk = negatives_config.get("max_per_chunk", 2)
    negatives_task_type = negatives_config.get("task_type", "rules_qa")

    verification_config = config.get("verification", {}) or {}
    verification_enabled = verification_config.get("enabled", False)
    verification_threshold = verification_config.get("threshold", 4)
    verification_use_corrections = verification_config.get("use_corrections", True)
    verification_model = verification_config.get("model", model)
    verification_stats = VerificationStats() if verification_enabled else None

    dedup_config = config.get("deduplication", {}) or {}
    dedup_enabled = dedup_config.get("enabled", False)
    dedup_threshold = dedup_config.get("similarity_threshold", 0.85)
    dedup_model = dedup_config.get("model", "all-MiniLM-L6-v2")
    dedup_cross_chunk = dedup_config.get("cross_chunk", True)
    deduplicator = None
    if dedup_enabled and dedup_cross_chunk:
        deduplicator = RunningDeduplicator(
            threshold=dedup_threshold,
            model_name=dedup_model,
        )

    # Track negative example counts for ratio management
    positive_count = 0
    negative_count = 0

    # Multi-turn conversation config
    multiturn_config = config.get("multiturn", {}) or {}
    multiturn_enabled = multiturn_config.get("enabled", False)
    multiturn_ratio = multiturn_config.get("ratio", 0.20)
    multiturn_min_turns = multiturn_config.get("min_turns", 2)
    multiturn_max_turns = multiturn_config.get("max_turns", 3)
    multiturn_task_types = multiturn_config.get("task_types") or task_types
    multiturn_stats = MultiturnStats() if multiturn_enabled else None
    multiturn_count = 0

    # Difficulty stratification config
    difficulty_config = config.get("difficulty", {}) or {}
    difficulty_enabled = difficulty_config.get("enabled", False)
    difficulty_distribution = difficulty_config.get("distribution", DEFAULT_DISTRIBUTION)
    difficulty_overrides = difficulty_config.get("overrides", {}) or {}
    difficulty_stats = DifficultyStats() if difficulty_enabled else None

    tables_config = config.get("tables", {}) or {}
    tables_enabled = tables_config.get("enabled", False)
    tables_min_rows = tables_config.get("min_rows", 4)
    tables_min_cols = tables_config.get("min_cols", 2)
    tables_max_rows = tables_config.get("max_rows", 12)
    tables_max_cols = tables_config.get("max_cols", 5)
    tables_max_pairs = tables_config.get("max_pairs", 5)
    tables_task_type = tables_config.get("task_type", "auto")

    coverage_config = config.get("coverage", {}) or {}
    coverage_enabled = coverage_config.get("enabled", False)
    coverage_min_text_len = coverage_config.get("min_text_len", 1400)
    coverage_max_pairs = coverage_config.get("max_pairs", 2)
    coverage_task_type = coverage_config.get("task_type", "auto")

    rag_config = config.get("rag_mode", {}) or {}
    rag_enabled = rag_config.get("enabled", False)
    rag_emit_messages = rag_config.get("emit_messages", True)
    rag_system_prompt = rag_config.get("system_prompt", RAG_SYSTEM_PROMPT)
    rag_citation_style = rag_config.get("citation_style", "(p. {page})")
    rag_format_overrides = rag_config.get("format_by_task_type", {}) or {}
    rag_default_format = rag_config.get("default_format", DEFAULT_RAG_FORMAT)

    random.shuffle(task_types)
    task_type_cycle = itertools.cycle(task_types)
    if start_index > 0:
        for _ in range(start_index):
            next(task_type_cycle)

    # Walkthrough mode: generate guided step-by-step conversations
    walkthrough_config = config.get("walkthrough", {}) or {}
    walkthrough_enabled = walkthrough_config.get("enabled", False)
    walkthrough_records = []
    if walkthrough_enabled:
        walkthrough_topic = walkthrough_config.get("topic", "the process")
        walkthrough_n = walkthrough_config.get("n_conversations", 10)
        walkthrough_turns = walkthrough_config.get("turns_per_conversation", 3)
        print(f"Generating {walkthrough_n} walkthrough conversations for: {walkthrough_topic}")

        walkthrough_records = generate_walkthrough_series(
            chunks=chunks,
            prompt_config=prompt_config,
            walkthrough_topic=walkthrough_topic,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            n_conversations=walkthrough_n,
            turns_per_conversation=walkthrough_turns,
            citation_style=rag_citation_style if rag_enabled else "(p. {page})",
            grounding_instructions="",
            format_instructions="",
            rag_system_prompt=rag_system_prompt if rag_enabled else "",
            repair_invalid_json=repair_invalid_json,
            invalid_log_path=invalid_response_log,
            warning_limiter=warning_limiter,
        )
        print(f"Generated {len(walkthrough_records)} walkthrough conversations")

    def resolve_task_type(config_value: str, fallback: str) -> str:
        if not config_value or config_value == "auto":
            return fallback
        if config_value in task_types:
            return config_value
        return fallback

    def format_instructions_for(task_type: str) -> str:
        if task_type in rag_format_overrides:
            return rag_format_overrides[task_type]
        return RAG_FORMATS.get(task_type, rag_default_format)

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

    # Write walkthrough records first if we have them
    walkthrough_written = 0
    if walkthrough_records and write_mode == "w":
        with open(output_path, "w", encoding="utf-8") as wf:
            for record in walkthrough_records:
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                walkthrough_written += 1
        write_mode = "a"  # Append remaining records
        count = walkthrough_written
        total_samples = walkthrough_written
        print(f"Wrote {walkthrough_written} walkthrough records")

    with open(output_path, write_mode, encoding="utf-8") as f:
        def write_record(
            pair: dict,
            fallback_task_type: str,
            source_page: int,
            context_block: str,
            context_pages: list,
            answer_format: str,
        ) -> bool:
            nonlocal count, total_samples, long_samples
            if max_samples is not None and count >= max_samples:
                return False

            missing = []
            if "instruction" not in pair:
                missing.append("instruction")
            if "output" not in pair:
                missing.append("output")
            if missing:
                print(f"Warning: Skipping malformed item on page {source_page}; missing {missing}.")
                return False

            pair_task_type = pair.get("task_type", fallback_task_type)
            if pair_task_type not in task_types:
                print(
                    f"Warning: Invalid task_type '{pair_task_type}' on page {source_page}; "
                    f"forcing to '{fallback_task_type}'."
                )
                pair_task_type = fallback_task_type

            output_text = pair["output"]
            if rag_enabled and pair.pop("_force_citation", False):
                citation = rag_citation_style.format(page=source_page)
                if "Citations:" not in output_text:
                    output_text = f"{output_text}\n\nCitations: {citation}"

            record = {
                "instruction": pair["instruction"],
                "input": context_block if rag_enabled else "",
                "output": output_text,
                "task_type": pair_task_type,
                "source_page": source_page,
            }
            if rag_enabled:
                record["context"] = context_block
                record["citations"] = context_pages
                record["answer_format"] = answer_format
                # Optional richer provenance when chunk artifacts are used
                if isinstance(chunk.get("doc_id"), str):
                    record["source_doc_id"] = chunk.get("doc_id")
                if isinstance(chunk.get("chunk_id"), str):
                    record["source_chunk_id"] = chunk.get("chunk_id")
                if isinstance(chunk.get("page_start"), int) and isinstance(chunk.get("page_end"), int):
                    record["source_page_start"] = chunk.get("page_start")
                    record["source_page_end"] = chunk.get("page_end")
                if rag_emit_messages:
                    record["messages"] = [
                        {"role": "system", "content": rag_system_prompt},
                        {
                            "role": "user",
                            "content": f"{pair['instruction']}\n\nContext:\n{context_block}",
                        },
                        {"role": "assistant", "content": output_text},
                    ]
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
            return True

        for order_pos in range(start_index, len(order)):
            if max_samples is not None and count >= max_samples:
                break

            chunk = chunks[order[order_pos]]
            prev_chars = context_config.get("prev_chars", 0)
            next_chars = context_config.get("next_chars", 0)
            prev_text = ""
            next_text = ""
            prev_page = None
            next_page = None
            # IMPORTANT: prev/next must be based on document adjacency, not shuffled sampling order.
            if prev_chars > 0:
                prev_idx = prev_by_idx.get(order[order_pos])
                if prev_idx is not None:
                    prev_chunk = chunks[prev_idx]
                    prev_text = prev_chunk.get("text", "")[-prev_chars:]
                    prev_page = prev_chunk.get("page") or prev_chunk.get("page_start")
            if next_chars > 0:
                next_idx = next_by_idx.get(order[order_pos])
                if next_idx is not None:
                    next_chunk = chunks[next_idx]
                    next_text = next_chunk.get("text", "")[:next_chars]
                    next_page = next_chunk.get("page") or next_chunk.get("page_start")

            analysis = analyze_text(chunk["text"])
            page = chunk.get("page")
            if page is None and isinstance(chunk.get("page_start"), int):
                page = int(chunk["page_start"])
            if page is None:
                page = "unknown"
            table_like = analysis["table_like"]
            if table_like:
                table_like_pages += 1

            table_pairs = []
            if tables_enabled and table_like:
                default_table_type = task_types[0] if task_types else "rules_qa"
                table_pairs = extract_table_pairs(
                    chunk["text"],
                    task_type=resolve_task_type(tables_task_type, default_table_type),
                    min_rows=tables_min_rows,
                    min_cols=tables_min_cols,
                    max_rows=tables_max_rows,
                    max_cols=tables_max_cols,
                    max_pairs=tables_max_pairs,
                )

            if analysis["low_signal"] and not table_pairs:
                skipped_low_signal += 1
                continue

            n_questions = suggest_questions(
                analysis["text_len"],
                analysis["table_like"],
                analysis["low_signal"],
            )
            if analysis["low_signal"]:
                n_questions = 0

            processed_pages += 1

            task_type = next(task_type_cycle)
            resolved_table_type = resolve_task_type(tables_task_type, task_type)
            if table_pairs:
                for pair in table_pairs:
                    pair["task_type"] = resolved_table_type
                    if rag_enabled:
                        pair["_force_citation"] = True

            context_pages = [page]
            # Use page span label if available
            if isinstance(chunk.get("page_start"), int) and isinstance(chunk.get("page_end"), int):
                label = f"{chunk['page_start']}-{chunk['page_end']}"
            else:
                label = str(page)
            context_parts = [f"[Page {label}]\n{chunk['text']}"]
            if prev_text:
                context_pages.append(prev_page)
                context_parts.insert(0, f"[Page {prev_page} - excerpt]\n{prev_text}")
            if next_text:
                context_pages.append(next_page)
                context_parts.append(f"[Page {next_page} - excerpt]\n{next_text}")
            context_pages = [p for p in context_pages if p is not None]
            context_pages = list(dict.fromkeys(context_pages))
            combined_text = "\n\n".join(context_parts)

            grounding_instructions = ""
            format_instructions = ""
            if rag_enabled:
                page_list = ", ".join(str(p) for p in context_pages) or "unknown"
                grounding_instructions = (
                    "### Grounding Rules\n"
                    "- Use only the provided context.\n"
                    "- If the context does not contain the answer, say \"Not found in context.\" "
                    "and ask one concise clarifying question.\n"
                    f"- Cite sources using {rag_citation_style} where page is one of: {page_list}.\n"
                    "- Include a final 'Citations:' line.\n"
                )
                format_instructions = format_instructions_for(task_type)

            remaining = None if max_samples is None else max_samples - count

            if table_pairs:
                if remaining is not None:
                    table_pairs = table_pairs[:remaining]
                requested_questions += len(table_pairs)
                for pair in table_pairs:
                    if not write_record(
                        pair,
                        resolved_table_type,
                        page,
                        combined_text,
                        context_pages,
                        resolved_table_type,
                    ):
                        break

            if max_samples is not None and count >= max_samples:
                break

            if coverage_enabled and not analysis["low_signal"] and coverage_max_pairs > 0:
                if analysis["text_len"] >= coverage_min_text_len:
                    remaining = None if max_samples is None else max_samples - count
                    if remaining is None or remaining > 0:
                        coverage_count = coverage_max_pairs
                        if remaining is not None:
                            coverage_count = min(coverage_count, remaining)
                        requested_questions += coverage_count
                        coverage_type = resolve_task_type(coverage_task_type, task_type)
                        coverage_pairs = generate_coverage_pairs(
                            combined_text,
                            model=model,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            max_completion_tokens=max_completion_tokens,
                            task_type=coverage_type,
                            allowed_task_types=task_types,
                            n_questions=coverage_count,
                            repair_invalid_json=repair_invalid_json,
                            invalid_log_path=invalid_response_log,
                            warning_limiter=warning_limiter,
                            grounding_instructions=grounding_instructions,
                            format_instructions=format_instructions_for(coverage_type),
                        )
                        for pair in coverage_pairs:
                            if not write_record(
                                pair,
                                coverage_type,
                                page,
                                combined_text,
                                context_pages,
                                coverage_type,
                            ):
                                break

            if max_samples is not None and count >= max_samples:
                break

            extra_instructions = ""
            if table_like:
                if tables_enabled:
                    extra_instructions = (
                        "### Table Hint\n"
                        "If the text looks tabular, prioritize rules, definitions, or exceptions "
                        "instead of listing every row."
                    )
                else:
                    extra_instructions = (
                        "### Table Hint\n"
                        "If the text looks tabular, extract key rows into compact Q/A pairs."
                    )

            # Difficulty stratification - select difficulty for this chunk's questions
            difficulty_instructions = ""
            current_difficulty = None
            if difficulty_enabled:
                dist = get_distribution_for_task(task_type, difficulty_distribution, difficulty_overrides)
                current_difficulty = select_difficulty(dist, shuffle_seed, order_pos)
                difficulty_instructions = build_difficulty_prompt_section(current_difficulty)

            if n_questions > 0:
                remaining = None if max_samples is None else max_samples - count
                if remaining is not None:
                    n_questions = min(n_questions, remaining)
                if n_questions > 0:
                    requested_questions += n_questions
                    qa_pairs = generate_qa_pairs(
                        combined_text,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        max_completion_tokens=max_completion_tokens,
                        task_type=task_type,
                        allowed_task_types=task_types,
                        extra_instructions="\n\n".join(
                            part for part in [
                                extra_instructions,
                                difficulty_instructions,
                                grounding_instructions,
                                format_instructions,
                            ] if part
                        ),
                        n_questions=n_questions,
                        repair_invalid_json=repair_invalid_json,
                        invalid_log_path=invalid_response_log,
                        warning_limiter=warning_limiter,
                        prompt_config=prompt_config,
                    )

                    # Track difficulty for stats
                    if difficulty_enabled and current_difficulty and qa_pairs:
                        for _ in qa_pairs:
                            difficulty_stats.record(current_difficulty)

                    # Verification pass
                    if verification_enabled and qa_pairs:
                        qa_pairs, v_stats = verify_and_filter_pairs(
                            qa_pairs,
                            context=combined_text,
                            prompt_config=prompt_config,
                            model=verification_model,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            max_completion_tokens=max_completion_tokens,
                            threshold=verification_threshold,
                            use_corrections=verification_use_corrections,
                            warning_limiter=warning_limiter,
                        )
                        verification_stats.update(v_stats)

                    # Deduplication pass
                    if dedup_enabled and qa_pairs:
                        if deduplicator:
                            qa_pairs = deduplicator.add_and_filter(qa_pairs)
                        # If not cross-chunk, we'd do per-chunk dedup here

                    for pair in qa_pairs:
                        if not write_record(
                            pair,
                            task_type,
                            page,
                            combined_text,
                            context_pages,
                            task_type,
                        ):
                            break
                        positive_count += 1

            # Negative example generation - use ratio-based calculation
            if negatives_enabled:
                neg_needed = calculate_negative_count(
                    current_positive_count=positive_count,
                    current_negative_count=negative_count,
                    target_ratio=negatives_ratio,
                    max_per_chunk=negatives_max_per_chunk,
                )
                if neg_needed > 0:
                    remaining = None if max_samples is None else max_samples - count
                    if remaining is None or remaining > 0:
                        neg_count = min(neg_needed, remaining or neg_needed)
                        neg_task = negatives_task_type if negatives_task_type != "auto" else task_type
                        neg_pairs = generate_negative_pairs(
                            combined_text,
                            prompt_config=prompt_config,
                            model=model,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            max_completion_tokens=max_completion_tokens,
                            task_type=neg_task,
                            n_questions=neg_count,
                            repair_invalid_json=repair_invalid_json,
                            invalid_log_path=invalid_response_log,
                            warning_limiter=warning_limiter,
                        )
                        for pair in neg_pairs:
                            pair.pop("_is_negative", None)  # Remove internal marker
                            if not write_record(
                                pair,
                                neg_task,
                                page,
                                combined_text,
                                context_pages,
                                neg_task,
                            ):
                                break
                            negative_count += 1

            # Multi-turn conversation generation
            if (
                multiturn_enabled
                and task_type in multiturn_task_types
                and should_generate_multiturn(order_pos, multiturn_ratio, shuffle_seed)
                and not analysis["low_signal"]
            ):
                remaining = None if max_samples is None else max_samples - count
                if remaining is None or remaining > 0:
                    n_turns = select_turn_count(
                        multiturn_min_turns,
                        multiturn_max_turns,
                        shuffle_seed,
                        order_pos,
                    )
                    conversation = generate_multiturn_conversation(
                        combined_text,
                        prompt_config=prompt_config,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        max_completion_tokens=max_completion_tokens,
                        task_type=task_type,
                        n_turns=n_turns,
                        citation_style=rag_citation_style,
                        grounding_instructions=grounding_instructions,
                        format_instructions=format_instructions,
                        repair_invalid_json=repair_invalid_json,
                        invalid_log_path=invalid_response_log,
                        warning_limiter=warning_limiter,
                    )
                    if conversation:
                        multiturn_stats.record_success(conversation["turn_count"])
                        # Write multi-turn as a special record with full messages
                        mt_record = {
                            "instruction": conversation["messages"][0]["content"],
                            "output": conversation["messages"][-1]["content"],
                            "task_type": conversation["task_type"],
                            "source_page": page,
                            "is_multiturn": True,
                            "turn_count": conversation["turn_count"],
                            "topic_summary": conversation.get("topic_summary", ""),
                        }
                        if rag_enabled:
                            mt_record["context"] = combined_text
                            mt_record["citations"] = context_pages
                            mt_record["messages"] = [
                                {"role": "system", "content": rag_system_prompt},
                            ] + conversation["messages"]
                        f.write(json.dumps(mt_record, ensure_ascii=False) + "\n")
                        count += 1
                        total_samples += 1
                        multiturn_count += 1
                        pbar.update(1)
                    else:
                        multiturn_stats.record_failure()

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

    # Quality enhancement summaries
    if negatives_enabled:
        total_examples = positive_count + negative_count
        actual_ratio = negative_count / total_examples if total_examples > 0 else 0
        print("\n--- Negative Examples ---")
        print(f"Positive examples: {positive_count}")
        print(f"Negative examples: {negative_count}")
        print(f"Negative ratio: {actual_ratio:.1%} (target: {negatives_ratio:.1%})")

    if verification_enabled and verification_stats:
        verification_stats.print_summary()

    if dedup_enabled and deduplicator:
        deduplicator.stats.print_summary()

    if multiturn_enabled and multiturn_stats:
        multiturn_stats.print_summary()
        print(f"Multi-turn conversations: {multiturn_count} ({multiturn_count/max(1,total_samples):.1%} of total)")

    if difficulty_enabled and difficulty_stats:
        difficulty_stats.print_summary(target_distribution=difficulty_distribution)

    if walkthrough_written > 0:
        print("\n--- Walkthrough Summary ---")
        print(f"Walkthrough conversations: {walkthrough_written}")

    print(f"\nSuccessfully generated {total_samples} pairs. Saved to {output_path}")


if __name__ == "__main__":
    main()
