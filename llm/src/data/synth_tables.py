import re
from typing import Dict, List, Optional, Tuple


def split_table_line(line: str) -> List[str]:
    if "|" in line:
        cols = [c.strip() for c in line.split("|") if c.strip()]
    else:
        cols = [c.strip() for c in re.split(r"\s{2,}|\t+", line) if c.strip()]
    return cols


def detect_header(rows: List[List[str]]) -> Tuple[Optional[List[str]], List[List[str]]]:
    if len(rows) < 2:
        return None, rows

    first = rows[0]
    second = rows[1]

    def has_digit(cell: str) -> bool:
        return any(ch.isdigit() for ch in cell)

    first_digits = sum(has_digit(c) for c in first)
    second_digits = sum(has_digit(c) for c in second)
    if first_digits == 0 and second_digits > 0:
        return first, rows[1:]
    return None, rows


def extract_table_pairs(
    text: str,
    task_type: str,
    min_rows: int,
    min_cols: int,
    max_rows: int,
    max_cols: int,
    max_pairs: int,
) -> List[Dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    rows = []
    for line in lines:
        cols = split_table_line(line)
        if len(cols) >= min_cols:
            rows.append(cols)

    if len(rows) < min_rows:
        return []

    header, data_rows = detect_header(rows)
    if header:
        header = header[:max_cols]
    data_rows = data_rows[:max_rows]

    pairs: List[Dict[str, str]] = []
    seen_keys = set()

    for row in data_rows:
        cols = row[:max_cols]
        if len(cols) < min_cols:
            continue
        key = cols[0]
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        fields = []
        if header and len(header) >= len(cols):
            for name, value in zip(header, cols):
                if name.strip().lower() in ("name", "item", "entry"):
                    continue
                fields.append(f"{name}: {value}")
        else:
            for idx, value in enumerate(cols[1:], start=1):
                fields.append(f"Field {idx}: {value}")

        if not fields:
            continue

        instruction = f"What are the stats for {key}?"
        output = "; ".join(fields)
        pairs.append(
            {
                "instruction": instruction,
                "output": output,
                "task_type": task_type,
            }
        )
        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs
