"""
Multi-turn conversation generation for synthetic data.

Generates realistic 2-4 turn conversations that simulate how users
actually interact with an RPG assistant - with follow-up questions,
clarifications, and deeper exploration of topics.
"""

import json
from typing import Dict, List, Optional

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response
from .synth_llm import WarningLimiter
from .synth_prompts import PromptConfig

DEFAULT_MULTITURN_TEMPLATE = """
You are simulating a realistic conversation between a player/GM and an expert {topic} assistant.

### Context (Source Material)
{text}

### Task
Generate a {n_turns}-turn conversation that feels natural and educational.

### Conversation Structure
- **Turn 1**: User asks an initial question about something in the context
- **Turn 2**: Assistant answers with citations, user asks a natural follow-up
- **Turn 3+**: Continue naturally until the topic is thoroughly explored

### Requirements
1. Each assistant response must be grounded in the provided context
2. Follow-up questions should deepen understanding, not just repeat
3. Include citations in format {citation_style} for rule references
4. The conversation should feel like a real player asking a knowledgeable GM
5. Final assistant message should feel conclusive

### Question Types to Include
Based on task type "{task_type}", the conversation should explore:
{task_guidance}

{grounding_instructions}

{format_instructions}

### Output Format
Return a valid JSON object:
{{
  "messages": [
    {{"role": "user", "content": "initial question"}},
    {{"role": "assistant", "content": "answer with citations"}},
    {{"role": "user", "content": "follow-up question"}},
    {{"role": "assistant", "content": "deeper answer"}}
  ],
  "topic_summary": "brief description of what the conversation covers",
  "task_type": "{task_type}"
}}

Do not include markdown formatting. Return JSON only.
"""

TASK_GUIDANCE = {
    "rules_qa": (
        "- Initial: Ask about a rule, mechanic, or interaction\n"
        "- Follow-up: Ask about edge cases, exceptions, or how it interacts with other rules\n"
        "- Deepen: Explore tactical implications or common mistakes"
    ),
    "character_build": (
        "- Initial: Ask about a build choice or optimization question\n"
        "- Follow-up: Ask about tradeoffs, alternatives, or synergies\n"
        "- Deepen: Discuss playstyle implications or situational considerations"
    ),
    "scenario_seed": (
        "- Initial: Ask for scenario ideas or encounter concepts\n"
        "- Follow-up: Ask for complications, twists, or scaling options\n"
        "- Deepen: Explore how to adapt for different party compositions"
    ),
    "gm_guidance": (
        "- Initial: Ask about running a situation or adjudicating a rule\n"
        "- Follow-up: Ask about player expectations or common pitfalls\n"
        "- Deepen: Discuss how to handle edge cases or disputes"
    ),
    "lore": (
        "- Initial: Ask about setting, factions, or history\n"
        "- Follow-up: Ask about connections to other lore elements\n"
        "- Deepen: Explore how to incorporate into gameplay"
    ),
}


def generate_multiturn_conversation(
    text_chunk: str,
    prompt_config: PromptConfig,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    task_type: str,
    n_turns: int = 2,
    citation_style: str = "(p. {page})",
    grounding_instructions: str = "",
    format_instructions: str = "",
    repair_invalid_json: bool = True,
    invalid_log_path: Optional[str] = None,
    warning_limiter: Optional[WarningLimiter] = None,
) -> Optional[Dict]:
    """
    Generate a multi-turn conversation grounded in the provided context.
    
    Args:
        text_chunk: Source context for grounding
        prompt_config: Configured prompt templates
        model: LLM model to use
        task_type: Type of conversation to generate
        n_turns: Number of exchange turns (2 = 4 messages total)
        
    Returns:
        Dict with 'messages' list and metadata, or None if generation fails
    """
    task_guidance = TASK_GUIDANCE.get(task_type, TASK_GUIDANCE["rules_qa"])
    
    prompt = DEFAULT_MULTITURN_TEMPLATE.format(
        topic=prompt_config.topic,
        text=text_chunk,
        n_turns=n_turns,
        task_type=task_type,
        task_guidance=task_guidance,
        citation_style=citation_style,
        grounding_instructions=grounding_instructions,
        format_instructions=format_instructions,
    )

    response = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
        response_format=(
            {"type": "json_object"} if "gpt-4" in model or "gpt-3.5" in model else None
        ),
    )

    if not response.strip():
        msg = "Warning: Empty multi-turn response from model."
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        return None

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found")
        
        data = json.loads(clean[start:end])
        
        # Validate structure
        messages = data.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages must be a list with at least 2 entries")
        
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("each message must be a dict")
            if msg.get("role") not in ("user", "assistant"):
                raise ValueError(f"invalid role: {msg.get('role')}")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError("empty message content")
        
        # Ensure alternating roles starting with user
        expected_role = "user"
        for msg in messages:
            if msg["role"] != expected_role:
                raise ValueError(f"expected {expected_role}, got {msg['role']}")
            expected_role = "assistant" if expected_role == "user" else "user"
        
        # Ensure ends with assistant
        if messages[-1]["role"] != "assistant":
            raise ValueError("conversation must end with assistant")
        
        return {
            "messages": messages,
            "topic_summary": data.get("topic_summary", ""),
            "task_type": data.get("task_type", task_type),
            "turn_count": len(messages) // 2,
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        msg = f"Warning: Failed to parse multi-turn response: {e}"
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)
        
        if invalid_log_path:
            log_invalid_response(invalid_log_path, response)
        
        return None


def should_generate_multiturn(
    chunk_index: int,
    multiturn_ratio: float,
    seed: int = 1337,
) -> bool:
    """
    Deterministic check for whether to generate multi-turn for this chunk.
    """
    import hashlib
    
    hash_input = f"multiturn:{seed}:{chunk_index}".encode()
    hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    threshold = int(multiturn_ratio * 0xFFFFFFFF)
    
    return hash_val < threshold


def select_turn_count(min_turns: int, max_turns: int, seed: int, chunk_index: int) -> int:
    """
    Deterministically select number of turns for variety.
    """
    import hashlib
    
    hash_input = f"turns:{seed}:{chunk_index}".encode()
    hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    
    range_size = max_turns - min_turns + 1
    return min_turns + (hash_val % range_size)


class MultiturnStats:
    """Tracks multi-turn generation statistics."""

    def __init__(self):
        self.total_generated = 0
        self.total_failed = 0
        self.turn_counts: Dict[int, int] = {}

    def record_success(self, turn_count: int) -> None:
        self.total_generated += 1
        self.turn_counts[turn_count] = self.turn_counts.get(turn_count, 0) + 1

    def record_failure(self) -> None:
        self.total_failed += 1

    def summary(self) -> Dict:
        return {
            "total_generated": self.total_generated,
            "total_failed": self.total_failed,
            "turn_distribution": dict(self.turn_counts),
            "success_rate": (
                self.total_generated / (self.total_generated + self.total_failed)
                if (self.total_generated + self.total_failed) > 0
                else 0.0
            ),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n--- Multi-Turn Summary ---")
        print(f"Generated: {s['total_generated']}")
        print(f"Failed: {s['total_failed']}")
        print(f"Turn distribution: {s['turn_distribution']}")
        print(f"Success rate: {s['success_rate']:.1%}")

