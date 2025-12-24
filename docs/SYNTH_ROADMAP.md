# Synthetic Data Enhancement Roadmap

This document captures future improvements for the synthetic data generation pipeline. These enhancements were identified during a comprehensive pipeline review and are prioritized for future implementation.

---

## Completed Enhancements

### P0: Core Quality Fixes (Implemented Dec 2024)

- [x] **Configurable Prompt Templates** (`src/data/synth_prompts.py`)

  - Prompts now read from YAML config instead of hardcoded strings
  - Use `{topic}` placeholder for dynamic system/context customization
  - Config section: `prompts:` in `synthetic_generic.yaml`

- [x] **Negative Example Generation** (`src/data/synth_negatives.py`)
  - Generates "Not found in context" training examples for RAG grounding
  - Deterministic sampling based on `negatives.ratio` config
  - Config section: `negatives:` in `synthetic_generic.yaml`

### P1: Quality Assurance (Implemented Dec 2024)

- [x] **Answer Verification Pass** (`src/data/synth_verify.py`)

  - LLM-based 1-5 scoring of generated Q/A pairs
  - Automatic correction of low-scoring answers
  - Configurable threshold filtering
  - Config section: `verification:` in `synthetic_generic.yaml`

- [x] **Semantic Deduplication** (`src/data/synth_dedup.py`)
  - Embedding-based similarity filtering using sentence-transformers
  - Cross-chunk deduplication via `RunningDeduplicator`
  - Config section: `deduplication:` in `synthetic_generic.yaml`

---

### P2: Diversity Enhancements (Implemented Dec 2024)

- [x] **Multi-Turn Conversation Generation** (`src/data/synth_multiturn.py`)

  - Generates 2-4 turn conversations with natural follow-ups
  - Simulates realistic user interactions with clarification questions
  - Task-type aware conversation guidance
  - Config section: `multiturn:` in `synthetic_generic.yaml`

- [x] **Difficulty Stratification** (`src/data/synth_difficulty.py`)
  - Three levels: basic (recall), intermediate (synthesis), advanced (edge cases)
  - Weighted random selection with configurable distribution
  - Task-type specific overrides (e.g., lore gets more basic questions)
  - Config section: `difficulty:` in `synthetic_generic.yaml`

### User Experience Enhancements (Implemented Dec 2024)

- [x] **Quality Dashboard Report** (`src/data/synth_report.py`)

  - Comprehensive post-generation analysis
  - Sample distribution by task type, difficulty, multi-turn
  - Flagged issues (short outputs, missing citations, very long samples)
  - Representative examples from each category
  - Run with: `python -m src.data.synth_report --input <data.jsonl> --output report.md`

- [x] **RPG Evaluation Benchmark** (`src/training/evaluate_rpg.py`)

  - Measures: accuracy, grounding fidelity, citation accuracy, refusal calibration
  - Breakdown by task type and difficulty level
  - YAML-based evaluation set format
  - Create template: `python -m src.training.evaluate_rpg --create-template eval_set.yaml`

- [x] **Multi-RPG Template System** (`config/templates/`)
  - Pre-built templates for D&D 5e, Lancer, Blades in the Dark
  - Game-specific task types, personas, and answer formats
  - Difficulty distribution tuned per system
  - Base template for creating new game configs

---

## Future Improvements (P3)

### Adversarial Example Generation

**Priority**: P3 | **Effort**: Medium | **Impact**: Medium

Generate edge-case questions that test model robustness against tricky inputs.

**Rationale**: Production users ask ambiguous, misleading, or multi-part questions. Training should include these patterns.

**Example Types**:

1. **Ambiguous questions**: "How much damage does it do?" (no subject specified)
2. **Mixed premises**: Questions combining valid rules with invented mechanics
3. **Conditional questions**: "What if X, but also Y?"
4. **Multi-part with mixed answerability**: "Explain A, B, and C" where C isn't in context

**Implementation Sketch**:

```python
ADVERSARIAL_PROMPT = """
Generate {n} adversarial test questions based on this context.

Types to include:
1. Ambiguous: Questions missing key context that require clarification
2. Trick: Questions that sound valid but ask about things not in the text
3. Compound: Multi-part questions where some parts are unanswerable
4. Edge case: Questions about extreme or unusual rule applications

For each, provide:
- instruction: The tricky question
- output: Ideal handling (clarification request, partial answer, or boundary acknowledgment)
- adversarial_type: Which category this tests

Context:
{text}
"""
```

**Config Addition**:

```yaml
adversarial:
  enabled: true
  ratio: 0.05 # 5% of samples
  types:
    - ambiguous
    - trick
    - compound
    - edge_case
```

---

### Entity-Aware Coverage Generation

**Priority**: P3 | **Effort**: High | **Impact**: Medium

Two-phase generation that first extracts game mechanic entities, then generates targeted questions per entity.

**Rationale**: Current coverage generation may miss specific named abilities, items, or numeric rules. Explicit entity extraction ensures nothing important is skipped.

**Phase 1: Entity Extraction**

```python
ENTITY_EXTRACTION_PROMPT = """
Extract all game mechanics entities from this text. Be exhaustive.

Categories:
- ABILITY: Named abilities, actions, reactions (e.g., "Overcharge", "Boost")
- ITEM: Equipment, weapons, systems (e.g., "GMS Assault Rifle")
- STAT: Numeric values with context (e.g., "2 Heat", "1d6+2 damage")
- CONDITION: Status effects, states (e.g., "Stunned", "Prone")
- RULE: Named rules or subsystems (e.g., "Structure damage", "Overheating")
- REQUIREMENT: Prerequisites, restrictions (e.g., "requires LL2", "1/round")

Context:
{text}

Output JSON:
{
  "entities": [
    {"type": "ABILITY", "name": "Overcharge", "span": "exact text", "line": N},
    ...
  ]
}
"""
```

**Phase 2: Entity-Targeted Questions**

```python
ENTITY_QA_PROMPT = """
Generate a Q/A pair specifically about this game entity:

Entity: {entity_name} ({entity_type})
Relevant text: {entity_span}
Full context: {text}

The question must specifically ask about {entity_name}.
The answer must include all relevant details from the context.
"""
```

**Benefits**:

- Ensures coverage of every named mechanic
- Creates a verifiable mapping from entities → questions
- Enables entity-level coverage reports

---

### Answer Format Enforcement with Retry

**Priority**: P3 | **Effort**: Medium | **Impact**: Low

Structural validation of answers with automatic retry on format failures.

**Current State**: RAG formats define expected structure but aren't enforced.

**Implementation Sketch**:

```python
FORMAT_REQUIREMENTS = {
    "rules_qa": {
        "required_sections": ["Answer:", "Rules Reference:", "Citations:"],
        "optional_sections": ["Example:"],
    },
    "character_build": {
        "required_sections": ["Assumptions:", "Recommendations:", "Citations:"],
        "optional_sections": ["Tradeoffs:", "Next Questions:"],
    },
    # ...
}

def validate_answer_format(answer: str, task_type: str) -> tuple[bool, list[str]]:
    reqs = FORMAT_REQUIREMENTS.get(task_type, {})
    missing = [s for s in reqs.get("required_sections", []) if s not in answer]
    return len(missing) == 0, missing

def generate_with_format_retry(prompt, task_type, max_retries=2):
    for attempt in range(max_retries + 1):
        response = call_llm(prompt)
        valid, missing = validate_answer_format(response, task_type)
        if valid:
            return response
        if attempt < max_retries:
            prompt += f"\n\nYour previous answer was missing: {missing}. Please include all required sections."
    return response  # Return best effort after retries
```

---

## Implementation Notes

### Dependency Considerations

- **Semantic Deduplication** requires `sentence-transformers` (add to `requirements_synth.txt`)
- **Multi-turn** increases token usage ~3x per sample
- **Entity extraction** doubles LLM calls per chunk

### Testing Strategy

1. Add smoke tests for each new module in `tests/`
2. Create golden test files with expected outputs
3. Run full pipeline with `debug.max_samples: 20` before production runs

### Metrics to Track

After implementing enhancements, track:

- Deduplication rate (% filtered)
- Verification pass rate (% scoring ≥ threshold)
- Negative example ratio in final dataset
- Average turns per multi-turn conversation
- Entity coverage percentage

---

## References

- Original analysis: Pipeline review session (Dec 2024)
- Related: `docs/CONFIG.md` for configuration documentation
- Related: `AGENTS.md` for project context
