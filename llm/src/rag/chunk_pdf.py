import argparse
import os
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

try:
    import fitz  # pymupdf
except ImportError:
    raise SystemExit(
        "❌ Error: PyMuPDF (fitz) is not installed. Install via: pip install pymupdf"
    )

from .schema import RagChunk
from .text_norm import approx_token_count, join_section_path, normalize_for_id, sha256_hex, sha256_str


def _iter_text_spans(page_dict: dict) -> Iterable[Tuple[float, str]]:
    for block in page_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                size = float(span.get("size") or 0.0)
                yield size, text


def _estimate_body_font_size(doc: fitz.Document, max_pages: Optional[int]) -> float:
    sizes: List[float] = []
    pages = len(doc) if max_pages is None else min(len(doc), max_pages)
    for page_idx in range(pages):
        page = doc[page_idx]
        page_dict = page.get_text("dict")
        for size, text in _iter_text_spans(page_dict):
            # Filter out tiny fragments and probable headings (longer lines are usually body)
            if len(text) < 20:
                continue
            sizes.append(round(size, 1))
    if not sizes:
        return 10.0
    # Mode tends to be stable for body text in well-structured PDFs.
    common = Counter(sizes).most_common(1)
    if common:
        return float(common[0][0])
    return float(statistics.median(sizes))


def _heading_level(
    font_size: float,
    body_size: float,
    level_deltas: List[float],
) -> int:
    """
    Returns 1..N (1 is highest level).
    """
    delta = font_size - body_size
    if not level_deltas:
        return 1
    # level_deltas are ordered high->low thresholds
    for idx, threshold in enumerate(level_deltas, start=1):
        if delta >= threshold:
            return idx
    return len(level_deltas) + 1


def _looks_table_like(lines: List[str]) -> bool:
    if len(lines) < 5:
        return False
    hits = 0
    for line in lines:
        if "|" in line or "\t" in line:
            hits += 1
            continue
        # multiple spaces suggests columns
        if "  " in line:
            hits += 1
    return hits / max(1, len(lines)) >= 0.3


def _page_lines_preserve_layout(page: fitz.Page) -> List[str]:
    """
    Reconstruct lines with newlines preserved (important for headings + tables).
    """
    page_dict = page.get_text("dict")
    lines: List[str] = []
    for block in page_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        block_lines: List[str] = []
        for line in block.get("lines", []) or []:
            parts = []
            for span in line.get("spans", []) or []:
                txt = span.get("text") or ""
                if txt:
                    parts.append(txt)
            joined = "".join(parts).strip()
            if joined:
                block_lines.append(joined)
        if block_lines:
            lines.extend(block_lines)
            lines.append("")  # blank line between blocks
    # Trim trailing empties
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _detect_headings_for_page(
    page: fitz.Page,
    body_size: float,
    min_size_delta: float,
    max_heading_words: int,
) -> Dict[str, float]:
    """
    Return a map of exact line text -> max font size found in that line.
    Used to decide if a reconstructed line is a heading.
    """
    page_dict = page.get_text("dict")
    line_max_size: Dict[str, float] = {}
    for block in page_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            parts = []
            max_size = 0.0
            for span in line.get("spans", []) or []:
                txt = span.get("text") or ""
                if txt:
                    parts.append(txt)
                max_size = max(max_size, float(span.get("size") or 0.0))
            joined = "".join(parts).strip()
            if not joined:
                continue
            if max_size < body_size + min_size_delta:
                continue
            if max_heading_words and len(joined.split()) > max_heading_words:
                continue
            line_max_size[joined] = max(line_max_size.get(joined, 0.0), max_size)
    return line_max_size


def _build_chunks(
    doc: fitz.Document,
    doc_id: str,
    source: str,
    max_pages: Optional[int],
    max_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    include_section_prefix: bool,
    min_size_delta: float,
    level_deltas: List[float],
    max_heading_words: int,
) -> List[RagChunk]:
    body_size = _estimate_body_font_size(doc, max_pages=max_pages)

    section_stack: List[str] = []
    chunks: List[RagChunk] = []

    # Current chunk assembly state (per section)
    current_span_texts: List[str] = []
    current_span_ids: List[str] = []
    current_pages: List[int] = []
    current_section_path: List[str] = []
    chunk_index = 0

    def flush_chunk() -> None:
        nonlocal chunk_index
        if not current_span_texts:
            return
        text = "\n\n".join(t for t in current_span_texts if t.strip()).strip()
        if not text:
            return
        token_est = approx_token_count(text)
        if token_est < min_chunk_tokens:
            return

        page_start = min(current_pages) if current_pages else 1
        page_end = max(current_pages) if current_pages else page_start
        section_path = list(current_section_path)
        heading = section_path[-1] if section_path else None
        chunk_id = sha256_str(doc_id, *current_span_ids)

        prefixed = None
        if include_section_prefix and section_path:
            prefixed = f"Section: {join_section_path(section_path)}\n\n{text}"

        chunks.append(
            RagChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                source=source,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                heading=heading,
                chunk_index=chunk_index,
                text=text,
                text_prefixed=prefixed,
                span_ids=list(current_span_ids),
            )
        )
        chunk_index += 1

    def start_new_section(section_path: List[str]) -> None:
        nonlocal current_span_texts, current_span_ids, current_pages, current_section_path
        flush_chunk()
        current_span_texts = []
        current_span_ids = []
        current_pages = []
        current_section_path = list(section_path)

    def add_span(page_num: int, span_text: str, section_path: List[str]) -> None:
        nonlocal current_span_texts, current_span_ids, current_pages, current_section_path
        if current_section_path != section_path:
            start_new_section(section_path)

        norm = normalize_for_id(span_text)
        if not norm:
            return
        span_id = sha256_str(doc_id, str(page_num), "/".join(section_path), norm)

        # Decide whether adding this span would exceed max_tokens
        prospective_text = "\n\n".join(current_span_texts + [span_text]).strip()
        if current_span_texts and approx_token_count(prospective_text) > max_tokens:
            # Flush and carry overlap spans forward
            flush_chunk()

            if overlap_tokens > 0 and current_span_texts:
                # Keep tail spans until we reach overlap_tokens estimate
                kept_texts: List[str] = []
                kept_ids: List[str] = []
                kept_pages: List[int] = []
                running = 0
                for t, sid, pg in zip(
                    reversed(current_span_texts),
                    reversed(current_span_ids),
                    reversed(current_pages),
                ):
                    running += approx_token_count(t)
                    kept_texts.insert(0, t)
                    kept_ids.insert(0, sid)
                    kept_pages.insert(0, pg)
                    if running >= overlap_tokens:
                        break
                current_span_texts = kept_texts
                current_span_ids = kept_ids
                current_pages = kept_pages
            else:
                current_span_texts = []
                current_span_ids = []
                current_pages = []

        current_span_texts.append(span_text)
        current_span_ids.append(span_id)
        current_pages.append(page_num)

    pages = len(doc) if max_pages is None else min(len(doc), max_pages)
    for page_idx in range(pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        headings = _detect_headings_for_page(
            page,
            body_size=body_size,
            min_size_delta=min_size_delta,
            max_heading_words=max_heading_words,
        )

        lines = _page_lines_preserve_layout(page)
        page_table_like = _looks_table_like([ln for ln in lines if ln.strip()])

        paragraph_lines: List[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if not paragraph_lines:
                return
            span_text = "\n".join(paragraph_lines).strip()
            paragraph_lines = []
            if not span_text:
                return
            add_span(page_num, span_text, list(section_stack))

        for line in lines:
            if not line.strip():
                flush_paragraph()
                continue

            max_size = headings.get(line)
            if max_size is not None:
                # Found a heading line; finalize any paragraph before changing sections
                flush_paragraph()
                level = _heading_level(
                    font_size=max_size,
                    body_size=body_size,
                    level_deltas=level_deltas,
                )
                # Maintain stack depth
                if level <= 1:
                    section_stack = [line]
                else:
                    # level=2 means keep 1 parent, etc.
                    keep = max(0, level - 1)
                    section_stack = section_stack[:keep] + [line]
                start_new_section(list(section_stack))
                continue

            paragraph_lines.append(line)

        flush_paragraph()

        # If a page is strongly table-like, we bias toward chunk boundaries to avoid mixing with prose.
        if page_table_like:
            flush_chunk()

    flush_chunk()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create heading-aware, overlapping chunks for RAG.")
    parser.add_argument("--config", type=str, default="config/synthetic_generic.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    project_name = config.get("project_name", "default")
    dataset_tag = config.get("dataset_tag", "v1")
    path_vars = {"project_name": project_name, "dataset_tag": dataset_tag}

    ingest = config.get("ingest", {}) or {}
    pdf_path = ingest.get("pdf_path")
    if not pdf_path:
        raise SystemExit("Config missing ingest.pdf_path")
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found at: {pdf_path}")

    rag_ingest = config.get("rag_ingest", {}) or {}
    if not rag_ingest.get("enabled", False):
        print("rag_ingest.enabled is false; nothing to do.")
        return

    out_path_tmpl = rag_ingest.get("chunks_output_path")
    if not out_path_tmpl:
        raise SystemExit("Config missing rag_ingest.chunks_output_path")
    out_path = out_path_tmpl.format(**path_vars)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    debug = config.get("debug", {}) or {}
    max_pages = None
    if debug.get("enabled"):
        mp = debug.get("max_pages")
        if isinstance(mp, int) and mp > 0:
            max_pages = mp
            print(f"Debug mode: limiting chunking to {max_pages} pages.")

    chunking = rag_ingest.get("chunking", {}) or {}
    max_tokens = int(chunking.get("max_tokens", 900))
    overlap_tokens = int(chunking.get("overlap_tokens", 150))
    min_chunk_tokens = int(chunking.get("min_chunk_tokens", 200))
    include_section_prefix = bool(rag_ingest.get("include_section_prefix", True))

    heading_cfg = rag_ingest.get("heading_detection", {}) or {}
    min_size_delta = float(heading_cfg.get("min_size_delta", 1.5))
    level_deltas_raw = heading_cfg.get("level_deltas", [4.0, 2.0])
    level_deltas = [float(x) for x in level_deltas_raw if isinstance(x, (int, float))]
    max_heading_words = int(heading_cfg.get("max_heading_words", 14))

    with open(pdf_path, "rb") as f:
        doc_id = sha256_hex(f.read())

    source = os.path.basename(pdf_path)
    print(f"Chunking {source} (doc_id={doc_id[:12]}...)")

    doc = fitz.open(pdf_path)
    chunks = _build_chunks(
        doc,
        doc_id=doc_id,
        source=source,
        max_pages=max_pages,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        include_section_prefix=include_section_prefix,
        min_size_delta=min_size_delta,
        level_deltas=level_deltas,
        max_heading_words=max_heading_words,
    )

    flush_every = int(ingest.get("flush_every", 50))
    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(chunk.model_dump_json(ensure_ascii=False) + "\n")
            if flush_every and i % flush_every == 0:
                f.flush()

    print(f"Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()


