import argparse
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional

import yaml

try:
    import fitz  # pymupdf
except ImportError:
    print("Error: PyMuPDF (fitz) is not installed.")
    print("Please run: pip install pymupdf")
    raise


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


def analyze_text(text: str) -> Dict[str, Any]:
    text_len = len(text)
    lines = [line for line in text.splitlines() if line.strip()]

    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)

    alpha_ratio = alpha / max(1, text_len)
    digit_ratio = digits / max(1, text_len)
    space_ratio = spaces / max(1, text_len)

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
        "line_count": len(lines),
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "space_ratio": space_ratio,
        "table_like": table_like,
        "low_signal": low_signal,
        "has_low_signal_keyword": has_low_signal_keyword,
    }


def suggest_questions(text_len: int, table_like: bool, low_signal: bool) -> int:
    if low_signal:
        return 0
    base = clamp(text_len // 800 + 2, 2, 6)
    if table_like and base < 6:
        base += 1
    return base


def resolve_pdf_path(pdf_path: Optional[str], config_path: Optional[str]) -> str:
    if pdf_path:
        return pdf_path
    if not config_path:
        raise ValueError("pdf_path is required if no config is provided.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    ingest_config = config.get("ingest", {}) or {}
    path = ingest_config.get("pdf_path")
    if not path:
        raise ValueError("Config does not contain ingest.pdf_path.")
    return path


def summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    values_sorted = sorted(values)
    return {
        "mean": float(sum(values_sorted) / len(values_sorted)),
        "median": float(statistics.median(values_sorted)),
        "p90": float(values_sorted[int(0.9 * (len(values_sorted) - 1))]),
        "p95": float(values_sorted[int(0.95 * (len(values_sorted) - 1))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDF pages for synthetic data planning.")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument("--config", type=str, help="Path to config YAML (optional).")
    parser.add_argument("--output_json", type=str, help="Optional JSON output path.")
    parser.add_argument("--max_pages", type=int, help="Optional cap on pages analyzed.")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf_path, args.config)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    if args.max_pages and args.max_pages > 0:
        page_count = min(page_count, args.max_pages)

    page_metrics = []
    question_counts = {}

    for page_idx in range(page_count):
        page = doc[page_idx]
        text = page.get_text()
        metrics = analyze_text(text)
        suggested = suggest_questions(
            metrics["text_len"],
            metrics["table_like"],
            metrics["low_signal"],
        )
        metrics.update(
            {
                "page": page_idx + 1,
                "suggested_questions": suggested,
            }
        )
        page_metrics.append(metrics)
        question_counts[suggested] = question_counts.get(suggested, 0) + 1

    text_lengths = [m["text_len"] for m in page_metrics]
    table_pages = [m for m in page_metrics if m["table_like"]]
    low_signal_pages = [m for m in page_metrics if m["low_signal"]]
    suggested_total = sum(m["suggested_questions"] for m in page_metrics)

    summary = {
        "pages_analyzed": page_count,
        "low_signal_pages": len(low_signal_pages),
        "table_like_pages": len(table_pages),
        "suggested_total_questions": suggested_total,
        "question_count_distribution": question_counts,
        "text_length_stats": summarize(text_lengths),
    }

    print("\nPDF Analysis Summary")
    print("--------------------")
    print(f"Pages analyzed: {summary['pages_analyzed']}")
    print(f"Low-signal pages: {summary['low_signal_pages']}")
    print(f"Table-like pages: {summary['table_like_pages']}")
    print(f"Suggested total questions: {summary['suggested_total_questions']}")
    print("Question count distribution:", summary["question_count_distribution"])
    print("Text length stats (chars):", summary["text_length_stats"])

    top_by_length = sorted(page_metrics, key=lambda m: m["text_len"], reverse=True)[:5]
    if top_by_length:
        print("\nTop 5 densest pages (by char length):")
        for item in top_by_length:
            print(
                f"  page {item['page']}: len={item['text_len']} "
                f"table_like={item['table_like']} suggested_q={item['suggested_questions']}"
            )

    top_tables = sorted(
        [m for m in page_metrics if m["table_like"]],
        key=lambda m: m["line_count"],
        reverse=True,
    )[:5]
    if top_tables:
        print("\nTop 5 table-like pages (by line count):")
        for item in top_tables:
            print(
                f"  page {item['page']}: lines={item['line_count']} "
                f"len={item['text_len']} suggested_q={item['suggested_questions']}"
            )

    if args.output_json:
        output = {
            "summary": summary,
            "pages": page_metrics,
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote analysis JSON to {args.output_json}")


if __name__ == "__main__":
    main()
