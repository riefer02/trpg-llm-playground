from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RagSpan(BaseModel):
    """
    Atomic unit used to build stable chunks.

    Spans should be paragraph-ish and stable across repeated extraction runs.
    """

    span_id: str = Field(description="Deterministic ID for this span.")
    page: int = Field(ge=1)
    section_path: List[str] = Field(default_factory=list)
    text: str
    block_type: Literal["text", "table", "list", "unknown"] = "text"


class RagChunk(BaseModel):
    """
    Chunk artifact written to JSONL (one object per line).
    """

    doc_id: str = Field(description="Deterministic ID for the source PDF.")
    chunk_id: str = Field(description="Deterministic ID for this chunk.")
    source: str = Field(description="Source filename (or friendly book name).")

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: List[str] = Field(default_factory=list)
    heading: Optional[str] = None

    chunk_index: int = Field(ge=0, description="Monotonic within-doc chunk sequence.")

    text: str
    text_prefixed: Optional[str] = None

    span_ids: List[str] = Field(default_factory=list)


