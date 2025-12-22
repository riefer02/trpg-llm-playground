# Config Reference (Concise)

## Synthetic Generation (`config/synthetic_generic.yaml`)
- `project_name`, `dataset_tag`: naming keys used in Drive paths and checkpoints.
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

## Validation
- `scripts/validate_synth.py`: validates JSON/JSONL for RAG-aware fields (messages/context/citations).
- `scripts/audit_synth.py`: samples records and reports basic grounding/format issues.
- `output.path`: synthetic JSONL output (use `{run_id}`).
- `output.run_id`: `auto` generates a timestamp.
- `output.append`: append to an existing output file.
- `output.flush_every`: flush interval while generating.
- `llm.*`: model and response settings; JSON repair toggles and log path.
- `logging.*`: HUD + warning volume controls.
- `task_types`: labels emitted per sample for dataset balancing/filters.

## Training (`config/rpg_finetune.yaml`)
- `model.base_model`: Unsloth base model.
- `dataset.train_path`: can be a glob to load multiple JSONL files.
- `training.output_dir`: LoRA output on Drive.
- `training.save_steps`: checkpoint frequency.
- `lora.*`: adapter config.

## Colab Notebooks
- `colab/run_synthetic_only.ipynb`: ingest + synthetic only.
- `colab/run_train_after_synth.ipynb`: training + eval only.
- `colab/run_pipeline.ipynb`: full pipeline.

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
