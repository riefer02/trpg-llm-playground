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

OLLAMA_MODEL = "lancer-expert"
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
    """Launch Gradio chat interface with enhanced UI."""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Install with:")
        print("   pip install gradio")
        print("\nFalling back to CLI mode...")
        cli_chat(index, chunks, embed_model, model_name)
        return

    # Custom CSS for a polished TTRPG aesthetic
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo+2:wght@300;400;500;600&display=swap');
    
    :root {
        --lancer-gold: #d4af37;
        --lancer-gold-dim: #b8942e;
        --lancer-cyan: #00d4ff;
        --lancer-cyan-dim: #0099bb;
        --lancer-dark: #0a0e14;
        --lancer-darker: #060a0f;
        --lancer-panel: #111820;
        --lancer-border: #1e2a38;
        --lancer-text: #e8ecf0;
        --lancer-text-dim: #8899aa;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, var(--lancer-darker) 0%, var(--lancer-dark) 50%, #0d1219 100%) !important;
        font-family: 'Exo 2', sans-serif !important;
        max-width: 1400px !important;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--lancer-border);
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(212,175,55,0.08) 0%, transparent 100%);
    }
    
    .main-header h1 {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: var(--lancer-gold) !important;
        text-shadow: 0 0 20px rgba(212,175,55,0.3);
        margin: 0 !important;
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: var(--lancer-text-dim) !important;
        margin-top: 0.5rem !important;
        font-size: 0.95rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid var(--lancer-cyan-dim);
        border-radius: 20px;
        color: var(--lancer-cyan);
        font-size: 0.8rem;
        font-family: 'Orbitron', monospace;
        margin-top: 0.75rem;
    }
    
    /* Chat container - target Gradio 6 structure */
    #chat-column {
        background: var(--lancer-panel) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 12px !important;
        padding: 0 !important;
        overflow: hidden;
    }
    
    /* Chatbot wrapper and inner elements */
    [data-testid="chatbot"],
    .chatbot,
    .wrap,
    .bubble-wrap {
        background: var(--lancer-panel) !important;
        border: none !important;
    }
    
    /* Target all possible chatbot containers */
    #chat-column > div,
    #chat-column [role="log"],
    #chat-column .overflow-y-auto {
        background: var(--lancer-panel) !important;
    }
    
    /* Message styling */
    .message,
    [data-testid="user"],
    [data-testid="bot"] {
        font-family: 'Exo 2', sans-serif !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        line-height: 1.6 !important;
    }
    
    /* User messages */
    [data-testid="user"],
    .user-message,
    .message.user {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(0,153,187,0.1) 100%) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        color: var(--lancer-text) !important;
    }
    
    /* Bot messages */
    [data-testid="bot"],
    .bot-message,
    .message.bot {
        background: linear-gradient(135deg, rgba(212,175,55,0.08) 0%, rgba(184,148,46,0.05) 100%) !important;
        border: 1px solid rgba(212,175,55,0.2) !important;
        color: var(--lancer-text) !important;
    }
    
    /* Input area */
    .input-row {
        background: var(--lancer-panel) !important;
        border-top: 1px solid var(--lancer-border) !important;
        padding: 1rem !important;
    }
    
    .input-row textarea {
        background: var(--lancer-darker) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 8px !important;
        color: var(--lancer-text) !important;
        font-family: 'Exo 2', sans-serif !important;
        padding: 0.75rem 1rem !important;
    }
    
    .input-row textarea:focus {
        border-color: var(--lancer-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.2) !important;
    }
    
    .input-row button {
        background: linear-gradient(135deg, var(--lancer-cyan) 0%, var(--lancer-cyan-dim) 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: var(--lancer-darker) !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .input-row button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,212,255,0.3) !important;
    }
    
    /* Sources panel */
    #sources-column {
        background: var(--lancer-panel) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .sources-header {
        font-family: 'Orbitron', monospace !important;
        color: var(--lancer-gold) !important;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--lancer-border);
    }
    
    .source-card {
        background: var(--lancer-darker) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
        transition: border-color 0.2s ease;
    }
    
    .source-card:hover {
        border-color: var(--lancer-gold-dim) !important;
    }
    
    .source-page {
        font-family: 'Orbitron', monospace;
        color: var(--lancer-cyan);
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .source-section {
        color: var(--lancer-text);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    .source-score {
        color: var(--lancer-text-dim);
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Examples styling */
    .examples-section {
        background: var(--lancer-panel) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-top: 1rem !important;
    }
    
    .examples-section button {
        background: var(--lancer-darker) !important;
        border: 1px solid var(--lancer-border) !important;
        border-radius: 6px !important;
        color: var(--lancer-text) !important;
        font-family: 'Exo 2', sans-serif !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .examples-section button:hover {
        border-color: var(--lancer-gold-dim) !important;
        background: rgba(212,175,55,0.1) !important;
    }
    
    /* Footer */
    footer {
        display: none !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--lancer-darker);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--lancer-border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--lancer-gold-dim);
    }
    
    /* Markdown in responses */
    .chatbot .bot strong {
        color: var(--lancer-gold) !important;
    }
    
    .chatbot .bot code {
        background: var(--lancer-darker) !important;
        color: var(--lancer-cyan) !important;
        padding: 0.1rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .chatbot .bot blockquote {
        border-left: 3px solid var(--lancer-gold) !important;
        padding-left: 1rem !important;
        margin: 0.5rem 0 !important;
        color: var(--lancer-text-dim) !important;
    }
    """

    def format_sources_html(sources):
        """Format sources as HTML cards."""
        if not sources:
            return """<div class="sources-header">📚 RETRIEVED SOURCES</div>
                     <p style="color: var(--lancer-text-dim); font-size: 0.85rem;">
                     Sources will appear here when you ask a question.</p>"""

        html = '<div class="sources-header">📚 RETRIEVED SOURCES</div>'
        for i, s in enumerate(sources, 1):
            section = s.get("section", "")
            section_html = (
                f'<div class="source-section">{section}</div>' if section else ""
            )
            score_pct = int(s["score"] * 100)
            html += f"""
            <div class="source-card">
                <div class="source-page">SOURCE {i} • PAGE {s["page"]}</div>
                {section_html}
                <div class="source-score">Relevance: {score_pct}%</div>
            </div>
            """
        return html

    def respond(message, history):
        """Process message and return response with sources."""
        if not message.strip():
            return history, format_sources_html([])

        answer, sources = chat_with_rag(message, index, chunks, embed_model, model_name)
        # Gradio 6 expects messages format with role/content dicts
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, format_sources_html(sources)

    def make_example_handler(example_text):
        """Create a handler for a specific example."""

        def handler(history):
            return respond(example_text, history)

        return handler

    # Build the interface
    with gr.Blocks(title="Lancer Rules Assistant") as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>⚔️ LANCER RULES ASSISTANT</h1>
            <p>Your AI-powered guide to the Lancer TTRPG ruleset</p>
            <div class="status-badge">● LOCAL • OLLAMA</div>
        </div>
        """)

        with gr.Row():
            # Main chat column
            with gr.Column(scale=3, elem_id="chat-column"):
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    show_label=False,
                    avatar_images=(None, "🤖"),
                )

                with gr.Row(elem_classes="input-row"):
                    msg = gr.Textbox(
                        placeholder="Ask about Lancer rules, mechs, combat, or lore...",
                        show_label=False,
                        scale=6,
                        container=False,
                    )
                    submit_btn = gr.Button("TRANSMIT", scale=1, variant="primary")

            # Sources sidebar
            with gr.Column(scale=1, elem_id="sources-column"):
                sources_display = gr.HTML(value=format_sources_html([]), label="")

        # Examples section
        with gr.Row(elem_classes="examples-section"):
            gr.Markdown("**Quick queries:**")
            for example in [
                "How does the LOCK ON action work?",
                "What are the different mech sizes?",
                "What is Overcharge?",
                "Explain heat and overheating",
            ]:
                ex_btn = gr.Button(example, size="sm")
                ex_btn.click(
                    fn=make_example_handler(example),
                    inputs=[chatbot],
                    outputs=[chatbot, sources_display],
                )

        # Wire up main inputs
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, sources_display],
        ).then(lambda: "", outputs=msg)

        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, sources_display],
        ).then(lambda: "", outputs=msg)

    print("\n🚀 Starting Gradio UI at http://localhost:7860")
    print(f"   Model: {model_name}")
    demo.launch(css=custom_css)


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

    # Check Ollama is running
    try:
        models_response = ollama.list()
    except Exception as e:
        print("❌ Error: Cannot connect to Ollama.")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   Error: {e}")
        sys.exit(1)

    # Check model exists
    # Handle both old dict format and new Model object format
    models_list = (
        models_response.get("models", [])
        if isinstance(models_response, dict)
        else getattr(models_response, "models", [])
    )
    model_names = []
    for m in models_list:
        name = m.get("name", "") if isinstance(m, dict) else getattr(m, "model", "")
        model_names.append(name.split(":")[0])

    if args.model not in model_names and args.model.split(":")[0] not in model_names:
        print(f"⚠️  Warning: Model '{args.model}' not found in Ollama.")
        print(f"   Available models: {model_names}")
        print("\n   To use a base model for testing: --model qwen2.5:7b")
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
