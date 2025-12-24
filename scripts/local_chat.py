#!/usr/bin/env python3
"""
Local TTRPG Rules Chat with Ollama + RAG

Usage:
    python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl
    python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl --ui
    python scripts/local_chat.py --model qwen2.5:7b --ui  # Test with base model
"""

import argparse
import json
import sys

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import ollama
except ImportError as e:
    print("❌ Missing dependencies. Install with:")
    print("   pip install -r requirements_local.txt")
    print(f"\n   Missing: {e}")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "lancer-rules"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

SYSTEM_PROMPT = """You are a Lancer RPG rules assistant. Answer questions about game mechanics, character building, and lore using ONLY the provided reference material. Always cite page numbers when referencing rules. If the answer is not in the provided context, say "I don't have information about that in the rules I can access." Do not make up rules."""


# ============================================================
# RAG Setup
# ============================================================


def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"✅ Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def build_index(chunks: list[dict], embed_model: SentenceTransformer):
    """Build FAISS index from chunks."""
    print("Building FAISS index...")
    texts = [c.get("text", c.get("content", "")) for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"✅ Index built with {index.ntotal} vectors")
    return index


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve top-k relevant chunks."""
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk.get("text", chunk.get("content", "")),
                "page": chunk.get("page", chunk.get("metadata", {}).get("page", "?")),
                "section": chunk.get("section", chunk.get("heading", "")),
                "score": float(scores[0][i]),
            }
        )
    return results


def format_context(retrieved: list[dict]) -> str:
    """Format retrieved chunks for the prompt."""
    parts = []
    for i, r in enumerate(retrieved, 1):
        section = f" [{r['section']}]" if r.get("section") else ""
        parts.append(f"[Source {i}, p.{r['page']}{section}]\n{r['text']}")
    return "\n\n".join(parts)


# ============================================================
# Chat Functions
# ============================================================


def chat_with_rag(
    query: str,
    index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    model_name: str,
) -> tuple[str, list[dict]]:
    """Process a query with RAG retrieval + Ollama generation."""

    # Retrieve context
    retrieved = retrieve(query, index, chunks, embed_model)
    context = format_context(retrieved)

    # Build prompt with context
    user_prompt = f"""Reference Material:
{context}

Question: {query}

Answer the question using ONLY the reference material above. Cite page numbers."""

    # Call Ollama
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response["message"]["content"]
    return answer, retrieved


def cli_chat(
    index, chunks: list[dict], embed_model: SentenceTransformer, model_name: str
):
    """Simple CLI chat loop."""
    print("\n" + "=" * 60)
    print(f"Lancer Rules Assistant (Local) - Model: {model_name}")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        print("\n🔍 Searching rules...")
        answer, sources = chat_with_rag(query, index, chunks, embed_model, model_name)

        print(f"\n🤖 Assistant: {answer}")
        print("\n--- Sources ---")
        for i, s in enumerate(sources, 1):
            section = f" - {s['section']}" if s.get("section") else ""
            print(f"  [{i}] p.{s['page']}{section} (score: {s['score']:.2f})")


def gradio_ui(
    index, chunks: list[dict], embed_model: SentenceTransformer, model_name: str
):
    """Launch Gradio chat interface."""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Install with:")
        print("   pip install gradio")
        print("\nFalling back to CLI mode...")
        cli_chat(index, chunks, embed_model, model_name)
        return

    def respond(message, history):
        answer, sources = chat_with_rag(message, index, chunks, embed_model, model_name)

        # Append sources
        source_text = "\n\n---\n**Sources:**\n"
        for i, s in enumerate(sources, 1):
            section = f" - {s['section']}" if s.get("section") else ""
            source_text += f"- [{i}] p.{s['page']}{section} (score: {s['score']:.2f})\n"

        return answer + source_text

    demo = gr.ChatInterface(
        respond,
        title="🤖 Lancer Rules Assistant (Local)",
        description=f"Ask questions about Lancer RPG rules. Running locally with Ollama ({model_name}).",
        examples=[
            "How does the LOCK ON action work?",
            "What are the different mech sizes?",
            "How do I calculate my mech's HP?",
            "What is Overcharge?",
            "Explain the difference between kinetic and energy damage.",
        ],
    )

    print(f"\n🚀 Starting Gradio UI at http://localhost:7860")
    print(f"   Model: {model_name}")
    demo.launch()


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Local TTRPG Rules Chat with Ollama + RAG"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="dataset/lancer_v1_chunks.jsonl",
        help="Path to chunks JSONL file",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio web UI instead of CLI",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    global TOP_K
    TOP_K = args.top_k

    # Check Ollama is running
    try:
        models_response = ollama.list()
    except Exception as e:
        print("❌ Error: Cannot connect to Ollama.")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   Error: {e}")
        sys.exit(1)

    # Check model exists
    model_names = [m["name"].split(":")[0] for m in models_response.get("models", [])]
    if args.model not in model_names and args.model.split(":")[0] not in model_names:
        print(f"⚠️  Warning: Model '{args.model}' not found in Ollama.")
        print(f"   Available models: {model_names}")
        print(f"\n   To use a base model for testing: --model qwen2.5:7b")
        print(
            f"   To create your fine-tuned model: ollama create {args.model} -f Modelfile"
        )

        # Offer to continue with a different model
        if model_names:
            print(f"\n   Continuing with first available model: {model_names[0]}")
            args.model = model_names[0]
        else:
            print("\n   No models available. Pull one with: ollama pull qwen2.5:7b")
            sys.exit(1)

    print(f"✅ Using Ollama model: {args.model}")

    # Load chunks
    try:
        chunks = load_chunks(args.chunks)
    except FileNotFoundError:
        print(f"❌ Chunks file not found: {args.chunks}")
        print("   Generate chunks first with the synthetic data pipeline.")
        sys.exit(1)

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Build index
    index = build_index(chunks, embed_model)

    # Run chat
    if args.ui:
        gradio_ui(index, chunks, embed_model, args.model)
    else:
        cli_chat(index, chunks, embed_model, args.model)


if __name__ == "__main__":
    main()
