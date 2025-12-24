# Focused Synthetic Data Generation

Generate topic-specific synthetic data (like character creation walkthroughs) and combine it with your general dataset.

## Quick Start

```bash
# 1. Generate focused data
python -m src.data.generate_synthetic --config config/templates/lancer_character_walkthrough.yaml

# 2. Merge with general dataset
python scripts/merge_datasets.py \
    dataset/lancer_v1_synthetic.jsonl \
    dataset/lancer_character_walkthrough_*.jsonl \
    -o dataset/lancer_combined.jsonl \
    --dedup
```

## When to Use Focused Generation

Use this approach when you want:
- **Walkthrough conversations**: Step-by-step guides like "help me create a character"
- **Topic depth**: More examples for a specific area (combat, crafting, etc.)
- **Process coverage**: Ensure the model knows how to guide users through multi-step tasks

## Creating a Focused Config

### Step 1: Copy the Base Template

```bash
cp config/templates/lancer_character_walkthrough.yaml config/my_focused_topic.yaml
```

### Step 2: Configure Topic Filtering

The `topic_filter` section controls which chunks are used for generation:

```yaml
topic_filter:
  enabled: true
  
  # Match chunks where section_path or heading contains ANY of these
  section_keywords:
    - "character"
    - "creation"
    - "pilot"
  
  # Match chunks where text contains ANY of these (optional)
  text_keywords: []
  
  # If true, ALL keywords must match. If false, ANY keyword matches.
  match_all: false
  
  # Optional: Use semantic similarity to find relevant chunks
  semantic:
    enabled: true
    query: "How to create a character and pilot in Lancer"
    threshold: 0.25  # Lower = more inclusive (0-1)
    top_k: 50        # Max chunks to keep (null = no limit)
    model: "all-MiniLM-L6-v2"
```

**Filtering Logic:**
1. Section keywords filter first (if provided)
2. Text keywords filter second (if provided)
3. Semantic filter last (if enabled)

Each filter narrows down the chunks further.

### Step 3: Configure Walkthrough Mode (Optional)

For guided step-by-step conversations:

```yaml
walkthrough:
  enabled: true
  topic: "creating a pilot character"  # Description for the LLM
  n_conversations: 20                   # How many to generate
  turns_per_conversation: 3             # 3 = 6 messages total
```

Walkthroughs are generated first, then regular Q&A generation continues with the filtered chunks.

### Step 4: Adjust Generation Settings

For focused datasets, you typically want:

```yaml
# Higher multi-turn ratio for conversational depth
multiturn:
  enabled: true
  ratio: 0.60  # 60% multi-turn conversations

# Narrower task types
task_types:
  - "character_build"
  - "rules_qa"

# Beginner-friendly difficulty distribution
difficulty:
  enabled: true
  distribution:
    basic: 0.50
    intermediate: 0.40
    advanced: 0.10
```

## Merging Datasets

### Basic Merge

Concatenate multiple JSONL files:

```bash
python scripts/merge_datasets.py data1.jsonl data2.jsonl -o combined.jsonl
```

### Merge with Deduplication

Remove near-duplicate questions:

```bash
python scripts/merge_datasets.py data/*.jsonl -o combined.jsonl --dedup --threshold 0.85
```

- `--threshold`: Similarity threshold (0-1). Higher = stricter dedup.
- `--model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)

### Merge with Shuffling

Randomize order (good for training):

```bash
python scripts/merge_datasets.py data/*.jsonl -o combined.jsonl --shuffle --seed 42
```

### Full Example

```bash
# Merge general + walkthrough + combat-focused datasets, deduplicate, shuffle
python scripts/merge_datasets.py \
    dataset/lancer_general.jsonl \
    dataset/lancer_character_walkthrough.jsonl \
    dataset/lancer_combat_deep.jsonl \
    -o dataset/lancer_final.jsonl \
    --dedup --threshold 0.85 \
    --shuffle --seed 42
```

## Finding Good Keywords

To see what sections exist in your chunks:

```bash
# List unique section paths
python3 -c "
import json
sections = set()
with open('dataset/lancer_v1_chunks.jsonl') as f:
    for line in f:
        c = json.loads(line)
        sp = c.get('section_path', [])
        if sp:
            sections.add(' > '.join(str(s) for s in sp[:2]))
for s in sorted(sections):
    print(s)
"
```

To test your filter before running generation:

```bash
python3 -c "
import json
from src.data.synth_filter import filter_chunks_by_section

chunks = [json.loads(line) for line in open('dataset/lancer_v1_chunks.jsonl')]
filtered = filter_chunks_by_section(chunks, ['character', 'pilot', 'creation'])
print(f'Filtered: {len(filtered)} / {len(chunks)} chunks')
for c in filtered[:10]:
    print(f\"  - {c.get('heading', '')[:60]}\")
"
```

## Example Configs

### Character Creation Walkthrough

```yaml
topic_filter:
  enabled: true
  section_keywords: ["character", "pilot", "creation", "background", "skill", "talent"]
  semantic:
    enabled: true
    query: "How to create a character pilot including background and skills"
    threshold: 0.25

walkthrough:
  enabled: true
  topic: "creating a pilot character"
  n_conversations: 20
```

### Combat Deep-Dive

```yaml
topic_filter:
  enabled: true
  section_keywords: ["combat", "attack", "damage", "action", "reaction", "turn"]
  semantic:
    enabled: true
    query: "Combat rules attacks damage actions and tactical options"
    threshold: 0.30

walkthrough:
  enabled: false  # Combat is less walkthrough-y

multiturn:
  ratio: 0.40
  
task_types:
  - "rules_qa"
  - "gm_guidance"
```

### Lore Encyclopedia

```yaml
topic_filter:
  enabled: true
  section_keywords: ["history", "faction", "world", "setting", "lore", "union"]
  semantic:
    enabled: true
    query: "Setting lore history factions and world background"
    threshold: 0.25

walkthrough:
  enabled: false

task_types:
  - "lore"
  
difficulty:
  distribution:
    basic: 0.60      # "What is X?"
    intermediate: 0.35
    advanced: 0.05
```

## Output Format

Both general and focused datasets use the same JSONL format:

```json
{
  "instruction": "How do I create a pilot in Lancer?",
  "output": "To create a pilot, start by...",
  "task_type": "character_build",
  "source_page": 42,
  "is_multiturn": true,
  "is_walkthrough": true,
  "turn_count": 3,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

This consistency means you can freely mix datasets from different generation runs.

## Tips

1. **Start small**: Test with `n_samples: 10` before full generation
2. **Check coverage**: Use the filter test script to ensure you're capturing the right chunks
3. **Semantic threshold**: Start at 0.25-0.30, adjust based on results
4. **Dedup threshold**: 0.85 is a good default; lower catches more duplicates
5. **Label your datasets**: Use descriptive `dataset_tag` values like `character_walkthrough_v1`

