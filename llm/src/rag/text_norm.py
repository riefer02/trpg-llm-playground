import hashlib
import re
from typing import Iterable


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_WS_RE = re.compile(r"\s+")


def normalize_for_id(text: str) -> str:
    """
    Normalize text for deterministic IDs.

    Keep meaning-stable normalization (whitespace collapse + strip).
    """
    return _WS_RE.sub(" ", text).strip()


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def approx_token_count(text: str) -> int:
    """
    Cheap token estimate (~4 chars/token for English-ish text).
    """
    text = text or ""
    return max(1, len(text) // 4)


def join_section_path(section_path: Iterable[str]) -> str:
    return " > ".join([p for p in section_path if p])


