import json
import os
from typing import Dict, List


def log_invalid_response(log_path: str, response: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(response)
        f.write("\n\n---\n\n")


def load_chunks(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)
        if input_path.lower().endswith(".jsonl") or first_char != "[":
            chunks = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSONL line: {line[:80]}")
            return chunks
        return json.load(f)
