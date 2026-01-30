import argparse
import json
import os
from typing import Any, Dict, List

import yaml

from openai import OpenAI

from .text_norm import approx_token_count, sha256_str


def _load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _get_text_for_embedding(chunk: dict) -> str:
    return (chunk.get("text_prefixed") or chunk.get("text") or "").strip()


def _embed_batch(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    # Preserve order
    return [d.embedding for d in resp.data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from rag_ingest chunks.")
    parser.add_argument("--config", type=str, default="config/synthetic_generic.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    project_name = config.get("project_name", "default")
    dataset_tag = config.get("dataset_tag", "v1")
    path_vars = {"project_name": project_name, "dataset_tag": dataset_tag}

    rag_ingest = config.get("rag_ingest", {}) or {}
    rag_index = config.get("rag_index", {}) or {}
    if not rag_ingest.get("enabled", False):
        print("rag_ingest.enabled is false; cannot build index.")
        return
    if not rag_index.get("enabled", False):
        print("rag_index.enabled is false; nothing to do.")
        return

    chunks_path_tmpl = rag_ingest.get("chunks_output_path")
    if not chunks_path_tmpl:
        raise SystemExit("Config missing rag_ingest.chunks_output_path")
    chunks_path = chunks_path_tmpl.format(**path_vars)
    if not os.path.exists(chunks_path):
        raise SystemExit(f"Chunks file not found at: {chunks_path}. Run chunk_pdf first.")

    faiss_cfg = rag_index.get("faiss", {}) or {}
    out_dir_tmpl = faiss_cfg.get("index_output_dir")
    if not out_dir_tmpl:
        raise SystemExit("Config missing rag_index.faiss.index_output_dir")
    out_dir = out_dir_tmpl.format(**path_vars)
    os.makedirs(out_dir, exist_ok=True)

    emb_cfg = rag_index.get("embedding", {}) or {}
    provider = emb_cfg.get("provider", "openai")
    if provider != "openai":
        raise SystemExit(f"Unsupported embedding provider: {provider}")
    model = emb_cfg.get("model", "text-embedding-3-small")
    batch_size = int(emb_cfg.get("batch_size", 64))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set, but rag_index.enabled is true.\n"
            "Set it in Colab Secrets (or env var) before running build_index."
        )

    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing FAISS/numpy dependencies. Install with: pip install -r requirements_rag.txt\n"
            f"Import error: {e}"
        )

    client = OpenAI(api_key=api_key)

    chunks = _load_jsonl(chunks_path)
    if not chunks:
        raise SystemExit("No chunks found.")

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for c in chunks:
        txt = _get_text_for_embedding(c)
        if not txt:
            continue
        texts.append(txt)
        metas.append(
            {
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "section_path": c.get("section_path") or [],
            }
        )

    print(f"Embedding {len(texts)} chunks with {model} (batch_size={batch_size})...")
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(_embed_batch(client, model=model, texts=batch))

    mat = np.array(vectors, dtype="float32")
    # Cosine similarity via inner product on normalized vectors.
    faiss.normalize_L2(mat)
    dim = mat.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    index_id = sha256_str(project_name, dataset_tag, model)
    index_path = os.path.join(out_dir, f"{project_name}_{dataset_tag}_{index_id[:12]}.index")
    meta_path = os.path.join(out_dir, f"{project_name}_{dataset_tag}_{index_id[:12]}_meta.jsonl")
    cfg_path = os.path.join(out_dir, f"{project_name}_{dataset_tag}_{index_id[:12]}_index_config.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for row_id, meta in enumerate(metas):
            f.write(json.dumps({"row_id": row_id, **meta}, ensure_ascii=False) + "\n")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_name": project_name,
                "dataset_tag": dataset_tag,
                "embedding_model": model,
                "chunks_path": chunks_path,
                "index_path": index_path,
                "meta_path": meta_path,
                "vector_dim": int(dim),
                "count": int(index.ntotal),
            },
            f,
            indent=2,
        )

    approx_tokens = sum(approx_token_count(t) for t in texts)
    print(f"Saved FAISS index: {index_path}")
    print(f"Saved meta JSONL: {meta_path}")
    print(f"Indexed {index.ntotal} chunks (~{approx_tokens} est tokens embedded)")


if __name__ == "__main__":
    main()


