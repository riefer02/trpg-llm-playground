import argparse
import json
import os
import random
import sys


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)

        if path.lower().endswith(".jsonl") or first_char != "[":
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield idx, json.loads(line), None
                except json.JSONDecodeError as e:
                    yield idx, None, f"Invalid JSONL line: {e}"
            return

        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            yield 1, None, f"Invalid JSON: {e}"
            return
        if not isinstance(data, list):
            yield 1, None, "JSON root is not a list."
            return
        for idx, record in enumerate(data, start=1):
            yield idx, record, None


def sample_records(path: str, sample_size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    total = 0
    for idx, record, err in iter_records(path):
        if err or record is None:
            continue
        total += 1
        if len(sample) < sample_size:
            sample.append((idx, record))
            continue
        j = rng.randint(0, total - 1)
        if j < sample_size:
            sample[j] = (idx, record)
    return sample, total


def audit_record(idx: int, record: dict):
    issues = []
    warnings = []

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        issues.append("missing messages")
        return issues, warnings

    if messages[-1].get("role") != "assistant":
        issues.append("last message not assistant")

    assistant = messages[-1].get("content", "")
    if not isinstance(assistant, str) or not assistant.strip():
        issues.append("empty assistant content")

    context = record.get("context")
    if not isinstance(context, str) or not context.strip():
        issues.append("missing context")

    citations = record.get("citations")
    if "Citations:" not in assistant and not citations:
        issues.append("missing citations")

    if "Not found in context." in assistant and "?" not in assistant:
        warnings.append("no clarifying question after Not found in context")

    return issues, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample-audit synthetic JSONL for grounding quality.")
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL file.")
    parser.add_argument("--sample", type=int, default=50, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    sample, total = sample_records(args.input, args.sample, args.seed)
    if not sample:
        print("No records found to audit.")
        return 1

    issue_counts = {}
    warning_counts = {}

    for idx, record in sample:
        issues, warnings = audit_record(idx, record)
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for warning in warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

    print("Audit summary")
    print(f"Total records: {total}")
    print(f"Sampled: {len(sample)}")
    if issue_counts:
        print("Issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {issue}: {count}")
    else:
        print("Issues: none")
    if warning_counts:
        print("Warnings:")
        for warning, count in sorted(warning_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {warning}: {count}")
    else:
        print("Warnings: none")

    return 1 if issue_counts else 0


if __name__ == "__main__":
    sys.exit(main())
