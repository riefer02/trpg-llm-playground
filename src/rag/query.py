import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI


def _load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _embed_query(client: OpenAI, model: str, query: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding


def _format_section_path(section_path: Any) -> str:
    if not isinstance(section_path, list):
        return ""
    parts = [str(p).strip() for p in section_path if str(p).strip()]
    return " > ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a FAISS chunk index.")
    parser.add_argument("--config", type=str, default="config/synthetic_generic.yaml")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--show_text", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    project_name = config.get("project_name", "default")
    dataset_tag = config.get("dataset_tag", "v1")
    path_vars = {"project_name": project_name, "dataset_tag": dataset_tag}

    rag_ingest = config.get("rag_ingest", {}) or {}
    rag_index = config.get("rag_index", {}) or {}
    if not rag_index.get("enabled", False):
        print("rag_index.enabled is false; nothing to query.")
        return

    chunks_path_tmpl = rag_ingest.get("chunks_output_path")
    if not chunks_path_tmpl:
        raise SystemExit("Config missing rag_ingest.chunks_output_path")
    chunks_path = chunks_path_tmpl.format(**path_vars)
    if not os.path.exists(chunks_path):
        raise SystemExit(f"Chunks file not found at: {chunks_path}")

    faiss_cfg = rag_index.get("faiss", {}) or {}
    out_dir_tmpl = faiss_cfg.get("index_output_dir")
    if not out_dir_tmpl:
        raise SystemExit("Config missing rag_index.faiss.index_output_dir")
    out_dir = out_dir_tmpl.format(**path_vars)
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Index dir not found: {out_dir}")

    emb_cfg = rag_index.get("embedding", {}) or {}
    emb_model = emb_cfg.get("model", "text-embedding-3-small")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set, but rag_index.enabled is true.\n"
            "Set it in Colab Secrets (or env var) before running query."
        )

    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing FAISS/numpy dependencies. Install with: pip install -r requirements_rag.txt\n"
            f"Import error: {e}"
        )

    # Pick newest index+meta in dir (simple v1).
    index_files = sorted(
        [f for f in os.listdir(out_dir) if f.endswith(".index")],
        key=lambda p: os.path.getmtime(os.path.join(out_dir, p)),
        reverse=True,
    )
    if not index_files:
        raise SystemExit(f"No .index files found in {out_dir}")
    index_path = os.path.join(out_dir, index_files[0])

    meta_files = sorted(
        [f for f in os.listdir(out_dir) if f.endswith("_meta.jsonl")],
        key=lambda p: os.path.getmtime(os.path.join(out_dir, p)),
        reverse=True,
    )
    if not meta_files:
        raise SystemExit(f"No *_meta.jsonl files found in {out_dir}")
    meta_path = os.path.join(out_dir, meta_files[0])

    index = faiss.read_index(index_path)
    meta_rows = _load_jsonl(meta_path)
    meta_by_row = {int(m["row_id"]): m for m in meta_rows if "row_id" in m}

    chunks = _load_jsonl(chunks_path)
    chunk_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}

    client = OpenAI(api_key=api_key)
    qvec = _embed_query(client, model=emb_model, query=args.query)
    qmat = np.array([qvec], dtype="float32")
    faiss.normalize_L2(qmat)

    scores, idxs = index.search(qmat, int(args.top_k))
    print(f"Query: {args.query}")
    print(f"Index: {index_path}")
    print(f"Meta: {meta_path}")
    print("")

    for rank, (row_id, score) in enumerate(zip(idxs[0].tolist(), scores[0].tolist()), start=1):
        meta = meta_by_row.get(int(row_id), {})
        chunk_id = meta.get("chunk_id")
        chunk = chunk_by_id.get(chunk_id, {})
        section = _format_section_path(meta.get("section_path"))
        pages = f"p.{meta.get('page_start')}-{meta.get('page_end')}"
        print(f"{rank}. score={score:.4f} {pages}  {section}")
        if args.show_text:
            text = (chunk.get("text") or "").strip()
            if text:
                print(text)
                print("")


if __name__ == "__main__":
    main()


