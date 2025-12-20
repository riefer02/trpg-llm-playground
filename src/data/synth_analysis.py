import re
from typing import Dict, List

LOW_SIGNAL_KEYWORDS = [
    "all rights reserved",
    "isbn",
    "copyright",
    "credits",
    "printed in",
    "graphic design",
    "art direction",
    "layout",
    "www.",
    "http",
]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def is_table_like(lines: List[str]) -> bool:
    if len(lines) < 5:
        return False
    table_like_lines = sum(
        1 for line in lines if re.search(r"\s{2,}|\t|\|", line)
    )
    ratio = table_like_lines / max(1, len(lines))
    return ratio >= 0.3


def analyze_text(text: str) -> Dict[str, object]:
    text_len = len(text)
    lines = [line for line in text.splitlines() if line.strip()]
    alpha = sum(c.isalpha() for c in text)

    alpha_ratio = alpha / max(1, text_len)
    lower_text = text.lower()
    has_low_signal_keyword = any(k in lower_text for k in LOW_SIGNAL_KEYWORDS)
    table_like = is_table_like(lines)

    low_signal = (
        text_len < 200
        or (alpha_ratio < 0.5 and text_len < 1200)
        or (has_low_signal_keyword and text_len < 1200)
    )

    return {
        "text_len": text_len,
        "lines": lines,
        "table_like": table_like,
        "low_signal": low_signal,
    }


def suggest_questions(text_len: int, table_like: bool, low_signal: bool) -> int:
    if low_signal:
        return 0
    base = clamp(text_len // 800 + 2, 2, 6)
    if table_like and base < 6:
        base += 1
    return base
