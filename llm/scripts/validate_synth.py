import argparse
import json
import os
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


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"
    for msg in messages:
        if not isinstance(msg, dict):
            return "message is not an object"
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            return f"invalid role: {role}"
        if not isinstance(content, str) or not content.strip():
            return f"empty content for role: {role}"
    if messages[-1].get("role") != "assistant":
        return "last message is not assistant"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate synthetic JSONL for RAG-aware training.")
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL file.")
    parser.add_argument("--require-messages", action="store_true", default=True)
    parser.add_argument("--no-require-messages", dest="require_messages", action="store_false")
    parser.add_argument("--require-context", action="store_true", default=True)
    parser.add_argument("--no-require-context", dest="require_context", action="store_false")
    parser.add_argument("--require-citations", action="store_true", default=True)
    parser.add_argument("--no-require-citations", dest="require_citations", action="store_false")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    total = 0
    errors = 0
    warnings = 0
    missing_messages = 0
    invalid_messages = 0
    missing_context = 0
    missing_citations = 0

    for idx, record, err in iter_records(args.input):
        if err:
            errors += 1
            print(f"[Line {idx}] {err}")
            continue
        if not isinstance(record, dict):
            errors += 1
            print(f"[Line {idx}] Record is not an object.")
            continue

        total += 1

        messages = record.get("messages")
        if messages is None:
            missing_messages += 1
            if args.require_messages:
                errors += 1
                print(f"[Line {idx}] Missing messages.")
            else:
                warnings += 1
            continue

        msg_error = validate_messages(messages)
        if msg_error:
            invalid_messages += 1
            errors += 1
            print(f"[Line {idx}] Invalid messages: {msg_error}")

        if args.require_context:
            context = record.get("context")
            if not isinstance(context, str) or not context.strip():
                missing_context += 1
                errors += 1
                print(f"[Line {idx}] Missing context.")

        if args.require_citations:
            assistant = messages[-1].get("content", "") if messages else ""
            citations = record.get("citations")
            if "Citations:" not in assistant and not citations:
                missing_citations += 1
                errors += 1
                print(f"[Line {idx}] Missing citations.")

    print("\nValidation summary")
    print(f"Records: {total}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    print(f"Missing messages: {missing_messages}")
    print(f"Invalid messages: {invalid_messages}")
    print(f"Missing context: {missing_context}")
    print(f"Missing citations: {missing_citations}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
