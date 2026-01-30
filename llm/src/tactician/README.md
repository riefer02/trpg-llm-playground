# Lancer Tactical AI - Prompt Design

This directory contains the prompt templates and utilities for the Lancer Tactical AI system.

## Architecture

The Tactical AI uses a two-step process:

1. **Combat State Serialization**: `state_serializer.py` converts the `MechCombatScenario` into a structured JSON format
2. **Prompt Construction**: `prompts.py` builds a complete LLM prompt combining system instructions with current combat state

## Prompt Design Decisions

### System Prompt Structure

The system prompt (`prompts/tactical_system.md`) is designed with several key sections:

#### 1. Lancer Combat Basics
- Includes essential mechanics (HP, Heat, Structure, Conditions, Action Economy)
- Written for an LLM with no prior Lancer knowledge
- Focuses on tactical implications rather than exhaustive rules

#### 2. Role-Specific Tactics
- Defines four combat roles: Striker, Defender, Artillery, Controller
- Each role has distinct tactical priorities
- Helps the AI make decisions consistent with mech capabilities
- Based on Lancer NPC role classifications (Striker, Defender, Artillery, Controller)

#### 3. Output Format Specification
- Requires JSON output with `action_id`, `target_id`, `reasoning`, and `confidence`
- Explicit validation rules to ensure output matches available actions
- Clear instructions for handling target-required vs targetless actions

#### 4. Few-Shot Examples
- Four examples covering different roles and situations
- Demonstrates proper reasoning structure
- Shows confidence scoring appropriate to situation uncertainty

### Design Rationale

#### Why JSON Output?
- Structured output enables reliable parsing
- Can be validated against available actions
- Extensible for future features (multi-action sequences)

#### Why Include Combat Basics?
- LLMs may not have Lancer-specific knowledge in context window
- Reduces hallucination of incorrect rules
- Ensures consistent tactical understanding across different model instances

#### Why Role-Specific Tactics?
- NPCs in Lancer have defined combat roles
- Different mech frames excel at different tactics
- Prevents "one-size-fits-all" decision making
- Allows for predictable, role-appropriate AI behavior

#### Why Confidence Scoring?
- Provides signal for fallback behavior (low confidence → random valid action)
- Helps debugging and analysis of AI decisions
- Future: could influence difficulty scaling

### Prompt Construction

The `build_tactical_prompt()` function:
1. Loads the system prompt from markdown file
2. Serializes combat state to JSON
3. Combines them with clear section dividers
4. Adds final instructions for output format

The `build_tactical_prompt_with_role()` variant:
- Adds explicit role emphasis
- Useful when NPC role is known from template

### Integration with Combat System

The prompt expects combat state in the format produced by `serialize_combat_state()`:

```python
from llm.src.tactician import serialize_combat_state, build_tactical_prompt

state = serialize_combat_state(scenario)
prompt = build_tactical_prompt(state)
```

### Testing

Prompt templates should be tested with:
- Valid combat states to ensure proper formatting
- Edge cases (empty scenario, single combatant)
- Role-specific variations

### Future Extensions

1. **Multi-turn Planning**: Extend output format to include sequence of actions
2. **Difficulty Scaling**: Adjust prompt aggressiveness based on difficulty level
3. **Personality Traits**: Add flavor text or stylistic preferences
4. **Learning from Outcomes**: Incorporate past decision outcomes into prompt

## Usage Example

```python
from llm.src.tactician import (
    serialize_combat_state,
    build_tactical_prompt_with_role,
)
from core.mech.combat_state import MechCombatScenario

# Get current combat scenario
scenario = MechCombatScenario(...)

# Serialize state
state = serialize_combat_state(scenario)

# Build prompt for artillery NPC
prompt = build_tactical_prompt_with_role(state, "artillery")

# Send to LLM and parse response
response = llm_complete(prompt)
action_choice = json.loads(response)
```