# RPG System Templates

This directory contains configuration templates for different RPG systems.
Each template provides game-specific customization for synthetic data generation.

## Available Templates

| Template | Game System | Description |
|----------|-------------|-------------|
| `lancer.yaml` | Lancer RPG | Tactical mech combat |
| `dnd5e.yaml` | D&D 5th Edition | Classic fantasy |
| `pathfinder2e.yaml` | Pathfinder 2e | Crunchy fantasy tactics |
| `cyberpunk_red.yaml` | Cyberpunk RED | Dystopian future |
| `blades.yaml` | Blades in the Dark | Heist-focused narrative |

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

