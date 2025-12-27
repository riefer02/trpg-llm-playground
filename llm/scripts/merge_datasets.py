#!/usr/bin/env python3
"""
Merge multiple synthetic datasets into one.

Usage:
    # Simple concatenation
    python scripts/merge_datasets.py dataset/general.jsonl dataset/walkthrough.jsonl -o dataset/combined.jsonl

    # With deduplication
    python scripts/merge_datasets.py dataset/*.jsonl -o dataset/combined.jsonl --dedup --threshold 0.85

    # Keep originals, just output merged
    python scripts/merge_datasets.py data1.jsonl data2.jsonl data3.jsonl -o merged.jsonl --dedup
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def load_jsonl(path: str) -> List[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at {path}:{line_num}: {e}")
    return records


def save_jsonl(records: List[dict], path: str) -> None:
    """Save records to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_schema(records: List[dict]) -> List[dict]:
    """
    Normalize all records to have the same columns.
    
    HuggingFace datasets requires consistent schema across all records.
    This adds missing columns with type-appropriate defaults.
    """
    if not records:
        return records
    
    # Collect all columns and infer their types from non-null values
    column_types = {}
    for record in records:
        for col, val in record.items():
            if val is not None and col not in column_types:
                column_types[col] = type(val)
    
    # Type-appropriate defaults
    def get_default(col):
        col_type = column_types.get(col)
        if col_type == str:
            return ""
        elif col_type == bool:
            return False
        elif col_type == int:
            return 0
        elif col_type == float:
            return 0.0
        elif col_type == list:
            return []
        elif col_type == dict:
            return {}
        else:
            return ""  # Default to empty string for unknown types
    
    # Normalize each record to have all columns
    all_columns = set(column_types.keys())
    normalized = []
    for record in records:
        norm_record = {}
        for col in all_columns:
            if col in record and record[col] is not None:
                norm_record[col] = record[col]
            else:
                norm_record[col] = get_default(col)
        normalized.append(norm_record)
    
    print(f"Normalized schema: {len(all_columns)} columns across all records")
    return normalized


def deduplicate_records(
    records: List[dict],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[dict]:
    """Remove duplicate records based on instruction similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print(
            "Error: sentence-transformers required for deduplication.\n"
            "Install with: pip install sentence-transformers"
        )
        sys.exit(1)

    if not records:
        return []

    print(f"Deduplicating {len(records)} records (threshold={threshold})...")

    model = SentenceTransformer(model_name)

    # Get instruction texts
    instructions = [r.get("instruction", "") for r in records]

    # Encode all instructions
    embeddings = model.encode(instructions, normalize_embeddings=True, show_progress_bar=True)

    # Find duplicates using pairwise similarity
    kept_indices = []
    kept_embeddings = []

    for i, emb in enumerate(embeddings):
        is_duplicate = False
        for kept_emb in kept_embeddings:
            similarity = float(np.dot(emb, kept_emb))
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_indices.append(i)
            kept_embeddings.append(emb)

    removed = len(records) - len(kept_indices)
    print(f"Removed {removed} duplicates ({removed / len(records) * 100:.1f}%)")

    return [records[i] for i in kept_indices]


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple merge
    python scripts/merge_datasets.py data1.jsonl data2.jsonl -o combined.jsonl

    # Merge with deduplication
    python scripts/merge_datasets.py data/*.jsonl -o combined.jsonl --dedup

    # Custom threshold (stricter)
    python scripts/merge_datasets.py data/*.jsonl -o combined.jsonl --dedup --threshold 0.90
        """,
    )
    parser.add_argument("inputs", nargs="+", help="Input JSONL files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by instruction similarity")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for deduplication (0-1, default: 0.85)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for deduplication",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    args = parser.parse_args()

    # Load all input files
    all_records = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"Warning: File not found: {input_path}")
            continue

        records = load_jsonl(input_path)
        print(f"Loaded {len(records)} records from {input_path}")

        # Tag source for traceability
        for record in records:
            if "source_dataset" not in record:
                record["source_dataset"] = path.stem

        all_records.extend(records)

    if not all_records:
        print("Error: No records loaded from input files.")
        sys.exit(1)

    print(f"\nTotal records before merge: {len(all_records)}")

    # Deduplicate if requested
    if args.dedup:
        all_records = deduplicate_records(
            all_records,
            threshold=args.threshold,
            model_name=args.model,
        )

    # Shuffle if requested
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(all_records)
        print(f"Shuffled records (seed={args.seed})")

    # Normalize schema so all records have the same columns
    all_records = normalize_schema(all_records)

    # Save output
    save_jsonl(all_records, args.output)
    print(f"\nSaved {len(all_records)} records to {args.output}")

    # Print summary by source
    sources = {}
    for r in all_records:
        src = r.get("source_dataset", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nRecords by source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()

