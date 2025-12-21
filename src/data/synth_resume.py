import hashlib
import json
import os
from typing import List, Optional


def build_signature(
    config: dict,
    input_path: str,
    output_path: str,
    task_types: List[str],
    shuffle: bool,
    shuffle_seed: int,
) -> str:
    input_stat = {}
    try:
        stat = os.stat(input_path)
        input_stat = {"size": stat.st_size, "mtime": stat.st_mtime}
    except OSError:
        input_stat = {}
    signature_payload = {
        "project_name": config.get("project_name"),
        "dataset_tag": config.get("dataset_tag"),
        "input_path": input_path,
        "input_stat": input_stat,
        "output_path": output_path,
        "task_types": task_types,
        "n_samples": config.get("n_samples"),
        "limits": config.get("limits"),
        "debug": config.get("debug"),
        "context": config.get("context"),
        "tables": config.get("tables"),
        "coverage": config.get("coverage"),
        "llm": config.get("llm"),
        "shuffle": shuffle,
        "shuffle_seed": shuffle_seed,
    }
    raw = json.dumps(signature_payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_checkpoint(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read checkpoint at {path}: {e}")
        return None


def save_checkpoint(path: str, data: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
