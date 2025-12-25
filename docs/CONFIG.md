# Config Reference (Concise)

## Synthetic Generation (`config/synthetic_generic.yaml`)

### Core Settings
- `project_name`, `dataset_tag`: naming keys used in Drive paths and checkpoints.
- `topic`: full name of the RPG system (used in prompts).
- `ingest.pdf_path`: source PDF on Drive.
- `ingest.raw_output_path`: JSONL output for extracted pages (persisted).
- `ingest.output_format`: `jsonl` recommended for streaming.
- `ingest.flush_every`: flush interval while ingesting.
- `debug.*`: small-sample limits for quick tests.
- `limits.enforce_max_samples`: if false, ignore `n_samples` cap.
- `generation.shuffle`, `generation.shuffle_seed`: deterministic page order.
- `resume.*`: enable/disable resume, checkpoint path, mismatch behavior.
- `tables.*`: optional table extraction pass (min/max rows/cols, max pairs, task type).
- `coverage.*`: optional coverage pass for dense pages (min text length, max pairs, task type).
- `rag_mode.*`: emit RAG-aware records (context + citations + optional `messages`) and enforce grounded formatting.
- `rag_ingest.*`: (optional) create heading-aware overlapping chunks with stable IDs for later retrieval.
- `rag_index.*`: (optional) build a FAISS index over `rag_ingest` chunks for semantic search.

### Quality Enhancement Settings (NEW)

#### Prompt Configuration
- `prompts.system_persona`: custom system prompt (use `{topic}` placeholder).
- `prompts.context_intro`: custom context introduction text.

#### Negative Examples
Generates "Not found in context" training examples for RAG grounding.
```yaml
negatives:
  enabled: true       # Enable negative example generation
  ratio: 0.12         # Target 12% of dataset
  max_per_chunk: 2    # Max negatives per chunk
  task_type: "rules_qa"  # Task type for negatives ("auto" = match chunk)
```

#### Answer Verification
LLM-based quality scoring and automatic correction.
```yaml
verification:
  enabled: true       # Enable verification pass
  threshold: 4        # Min score (1-5) to keep
  use_corrections: true  # Use corrected answers for low scores
  model: "gpt-4o-mini"   # Optional: different model for verification
```

#### Semantic Deduplication
Removes similar questions using sentence embeddings.
```yaml
deduplication:
  enabled: true
  similarity_threshold: 0.85  # Cosine similarity threshold
  model: "all-MiniLM-L6-v2"   # Sentence transformer model
  cross_chunk: true           # Dedupe across entire dataset
```

#### Multi-Turn Conversations
Generates realistic 2-4 turn dialogues with follow-ups.
```yaml
multiturn:
  enabled: true
  ratio: 0.20          # 20% of samples
  min_turns: 2         # Minimum turns (user+assistant = 1 turn)
  max_turns: 3
  task_types:          # Which tasks get multi-turn
    - "rules_qa"
    - "character_build"
    - "gm_guidance"
```

#### Difficulty Stratification
Ensures diverse cognitive complexity in questions.
```yaml
difficulty:
  enabled: true
  distribution:
    basic: 0.30        # Direct factual recall
    intermediate: 0.50  # Synthesis and comparison
    advanced: 0.20      # Edge cases, complex reasoning
  overrides:           # Per-task-type overrides
    lore:
      basic: 0.50
      intermediate: 0.40
      advanced: 0.10
```

## Output & Logging
- `output.path`: synthetic JSONL output (use `{run_id}`).
- `output.run_id`: `auto` generates a timestamp.
- `output.append`: append to an existing output file.
- `output.flush_every`: flush interval while generating.
- `llm.*`: model and response settings; JSON repair toggles and log path.
- `logging.*`: HUD + warning volume controls.
- `task_types`: labels emitted per sample for dataset balancing/filters.

## Validation & Reporting
- `scripts/validate_synth.py`: validates JSON/JSONL for RAG-aware fields (messages/context/citations).
- `scripts/audit_synth.py`: samples records and reports basic grounding/format issues.
- `python -m src.data.synth_report`: generates quality dashboard markdown report.
- `python -m src.training.evaluate_rpg`: RPG-specific evaluation benchmark.

## Training (`config/rpg_finetune.yaml`)

### Model Settings
- `model.base_model`: Unsloth base model (e.g., `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`)
- `model.max_seq_length`: Max tokens per training example (reduce if OOM)
- `model.load_in_4bit`: Use 4-bit quantization (recommended)

### Dataset
- `dataset.train_path`: Path to training JSONL (can use `{project_name}`, `{dataset_tag}`)

### Training
- `training.output_dir`: LoRA output on Drive
- `training.save_steps`: Checkpoint frequency
- `training.max_steps`: Total training steps

### Memory Configuration (Critical!)

Training memory usage depends on batch size, sequence length, and LoRA rank.
If you hit **CUDA out of memory** errors, adjust these settings:

| GPU | VRAM | `per_device_train_batch_size` | `gradient_accumulation_steps` | Notes |
|-----|------|------------------------------|------------------------------|-------|
| T4 | 16GB | 2 | 8 | May need `max_seq_length: 2048` |
| A100 | 40GB | 4 | 4 | Safe default |
| A100 | 80GB | 8 | 2 | Faster training |

**Key principle**: Keep effective batch size ~16 for good convergence.
`effective_batch = per_device_batch × gradient_accumulation × num_gpus`

**OOM Troubleshooting** (in order of impact):
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to compensate
3. Reduce `model.max_seq_length`
4. Reduce `lora.r` (LoRA rank)

### LoRA Settings
- `lora.r`: Rank (higher = more expressive, more VRAM). T4: 16, A100: 32
- `lora.lora_alpha`: Usually set equal to `r`
- `lora.use_gradient_checkpointing`: Set to `"unsloth"` for memory efficiency

## Colab Notebooks
- `colab/run_synthetic_only.ipynb`: ingest + synthetic only.
- `colab/run_train_after_synth.ipynb`: training + eval only.
- `colab/run_pipeline.ipynb`: full pipeline.

## Multi-RPG Templates (`config/templates/`)

Pre-built configurations for different game systems. Copy and customize:

```bash
cp config/templates/dnd5e.yaml config/my_config.yaml
```

| Template | System | Notes |
|----------|--------|-------|
| `_base.yaml` | Generic | Starting point for new systems |
| `dnd5e.yaml` | D&D 5th Edition | Includes `spell_lookup` task type |
| `lancer.yaml` | Lancer RPG | Includes `mech_stats` task type |
| `blades.yaml` | Blades in the Dark | Includes `npc_faction` task type |

Each template customizes:
- Task types specific to the system
- Answer format instructions per task type
- Difficulty distribution tuned to the system's complexity
- System-specific terminology and persona

## Optional RAG (Chunks + FAISS)

### Chunk artifact (recommended first step)
- Enable `rag_ingest.enabled: true`
- Run:
  - `python -m src.rag.chunk_pdf --config config/synthetic_generic.yaml`
- Output: `rag_ingest.chunks_output_path` (JSONL; one chunk per line with stable IDs).

### Build FAISS index (semantic search)
- Enable `rag_index.enabled: true`
- Install deps:
  - `pip install -r requirements_rag.txt`
- Run:
  - `python -m src.rag.build_index --config config/synthetic_generic.yaml`
- Query:
  - `python -m src.rag.query --config config/synthetic_generic.yaml --query "How does Overcharge work?" --top_k 5`
