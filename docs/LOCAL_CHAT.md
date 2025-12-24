# Local Chat with Ollama + RAG

Run your fine-tuned TTRPG rules assistant locally with full RAG retrieval.

## Quick Start

```bash
# Install dependencies
pip install -r requirements_local.txt

# Run with Web UI
python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl --ui
```

## Prerequisites

- **Ollama** installed ([ollama.ai/download](https://ollama.ai/download))
- **Python 3.10+**
- **8GB+ RAM** (16GB recommended)
- Your chunks file (generated during synthetic data pipeline)

## Installation

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download from [ollama.ai/download](https://ollama.ai/download)

### 2. Install Python Dependencies

```bash
cd trpg-llm-playground
pip install -r requirements_local.txt
```

### 3. Start Ollama Service

```bash
ollama serve
# Runs in background, or use the Ollama app on macOS
```

## Usage

### Test with Base Model (Before Training)

Validate your RAG pipeline before investing in fine-tuning:

```bash
# Pull a base model
ollama pull qwen2.5:7b

# Run chat with base model
python scripts/local_chat.py --model qwen2.5:7b --ui
```

### Run with Fine-Tuned Model

After training and GGUF export from Colab:

```bash
# Create Modelfile (in same directory as your .gguf file)
cat > Modelfile << 'EOF'
FROM ./lancer-rules-7b-q4_k_m.gguf

SYSTEM """You are a Lancer RPG rules assistant. Answer questions using the provided context. Always cite page numbers. If the answer is not in the context, say so."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Import into Ollama
ollama create lancer-rules -f Modelfile

# Verify
ollama list

# Run chat
python scripts/local_chat.py --model lancer-rules --ui
```

## Command Reference

```bash
# Web UI (recommended)
python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl --ui

# CLI mode
python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl

# Specify model
python scripts/local_chat.py --model qwen2.5:7b --ui

# Adjust retrieval (more/fewer sources)
python scripts/local_chat.py --top-k 5 --ui
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--chunks` | `dataset/lancer_v1_chunks.jsonl` | Path to RAG chunks file |
| `--model` | `lancer-rules` | Ollama model name |
| `--ui` | False | Launch Gradio web interface |
| `--top-k` | 3 | Number of chunks to retrieve |

## How It Works

```
User Question
     │
     ▼
┌─────────────┐
│ Embed Query │  (sentence-transformers)
└─────────────┘
     │
     ▼
┌─────────────┐
│ FAISS Search│  → Top-K relevant chunks
└─────────────┘
     │
     ▼
┌─────────────┐
│ Build Prompt│  Question + Retrieved Context
└─────────────┘
     │
     ▼
┌─────────────┐
│ Ollama Gen  │  → Answer with citations
└─────────────┘
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | SSD preferred |
| GPU | Optional | RTX 3060+ or M1 Mac |

### Performance by Hardware

| Setup | First Response | Subsequent |
|-------|----------------|------------|
| M1 Mac (16GB) | ~10s | ~2-3s |
| RTX 3060 (12GB) | ~5s | ~1-2s |
| CPU only (16GB) | ~30s | ~10s |

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"

```bash
# List available models
ollama list

# Pull a base model for testing
ollama pull qwen2.5:7b

# Or create your fine-tuned model
ollama create lancer-rules -f Modelfile
```

### "Out of memory"

- Close other applications
- Use a smaller model (3B instead of 7B)
- Use more aggressive quantization (Q4_0 instead of Q4_K_M)

### Slow first response

Normal! First query loads the model into memory (~10-30s). Keep Ollama running for fast subsequent queries.

## File Locations

After setup, your files should look like:

```
trpg-llm-playground/
├── dataset/
│   └── lancer_v1_chunks.jsonl    # RAG chunks
├── scripts/
│   └── local_chat.py             # Chat script
└── requirements_local.txt         # Dependencies

~/.ollama/models/                   # Ollama model storage
└── lancer-rules/                   # Your imported model
```

## Next Steps

1. **Compare base vs fine-tuned**: Test same questions with both
2. **Adjust retrieval**: Try `--top-k 2` or `--top-k 5`
3. **Improve training data**: Use chat failures to identify gaps
4. **Build desktop app**: Wrap in Tauri/Electron for distribution

