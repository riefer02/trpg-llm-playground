# Local Models Directory

This folder stores GGUF model files for local inference with Ollama.

## Setup Instructions

### 1. Download Your GGUF

After exporting from Colab, download the `.gguf` file from Google Drive:

- Look in: `My Drive/llm_experiments/outputs/`
- File pattern: `<model-name>-q4_k_m.gguf` (e.g., `qwen-lancer-7b-q4_k_m.gguf`)
- Place it in this `models/` folder

### 2. Update the Modelfile

Edit `Modelfile.lancer` and update the `FROM` line to match your filename:

```
FROM ./your-actual-filename.gguf
```

### 3. Create the Ollama Model

```bash
# From project root
ollama create lancer-expert -f models/Modelfile.lancer
```

### 4. Test It

```bash
# Quick test
ollama run lancer-expert "How does Overcharge work in Lancer?"

# Or use the full local chat with RAG
python scripts/local_chat.py --model lancer-expert
```

## Files

| File               | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `Modelfile.lancer` | Ollama config with system prompt (tracked in git) |
| `*.gguf`           | Model weights (gitignored - too large)            |

## Quantization Options

| Method   | File Size | Quality | Speed  |
| -------- | --------- | ------- | ------ |
| `q4_k_m` | ~4GB      | Good    | Fast   |
| `q8_0`   | ~8GB      | Better  | Medium |
| `f16`    | ~14GB     | Best    | Slower |

## Troubleshooting

### GGUF Export Fails with "bitsandbytes" Error

The Colab notebooks handle this by:

1. Reloading the model from HuggingFace in full precision (no bitsandbytes)
2. Saving a clean copy without quantization metadata
3. Converting to GGUF using llama.cpp

Make sure you're using the updated notebook cells and that you've pushed to HuggingFace first.

### "Model not found" in Ollama

Ensure the `FROM` path in `Modelfile.lancer` exactly matches your `.gguf` filename.
