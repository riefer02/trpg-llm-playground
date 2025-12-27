# RPG System Templates

This directory contains configuration templates for different RPG systems.
Each template provides game-specific customization for synthetic data generation.

## Available Templates

### System Templates

| Template             | Game System        | Description             |
| -------------------- | ------------------ | ----------------------- |
| `lancer.yaml`        | Lancer RPG         | Tactical mech combat    |
| `dnd5e.yaml`         | D&D 5th Edition    | Classic fantasy         |
| `pathfinder2e.yaml`  | Pathfinder 2e      | Crunchy fantasy tactics |
| `cyberpunk_red.yaml` | Cyberpunk RED      | Dystopian future        |
| `blades.yaml`        | Blades in the Dark | Heist-focused narrative |

### Focused/Walkthrough Templates

| Template                            | Purpose                                 |
| ----------------------------------- | --------------------------------------- |
| `lancer_character_walkthrough.yaml` | Guided character creation conversations |

## Usage

1. Copy a template to create your config:

   ```bash
   cp config/templates/dnd5e.yaml config/my_dnd_campaign.yaml
   ```

2. Customize paths and settings for your specific book/campaign

3. Run the pipeline:
   ```bash
   python -m src.data.generate_synthetic --config config/my_dnd_campaign.yaml
   ```

## Creating New Templates

Use `_base.yaml` as a starting point. Key customization areas:

1. **Topic & Project Name**: Set the game system name
2. **Task Types**: Add/remove types relevant to the system
3. **Prompts**: Customize persona and terminology
4. **Difficulty Overrides**: Adjust for system complexity
5. **Format Instructions**: Match the game's style

## Template Variables

Templates use these placeholders:

- `{topic}` - The game system name
- `{project_name}` - Short identifier for file naming
- `{dataset_tag}` - Version tag for the dataset

## Topic-Focused Data Generation

For focused datasets (like character creation walkthroughs), use `topic_filter`:

```yaml
topic_filter:
  enabled: true
  section_keywords: ["character", "creation", "pilot"]
  semantic:
    enabled: true
    query: "How to create a character"
    threshold: 0.25
```

Then enable `walkthrough` mode for guided conversations:

```yaml
walkthrough:
  enabled: true
  topic: "creating a pilot character"
  n_conversations: 20
  turns_per_conversation: 3
```

## Merging Datasets

Combine focused datasets with general ones:

```bash
# Simple merge
python scripts/merge_datasets.py general.jsonl walkthrough.jsonl -o combined.jsonl

# With deduplication
python scripts/merge_datasets.py *.jsonl -o combined.jsonl --dedup --threshold 0.85
```
