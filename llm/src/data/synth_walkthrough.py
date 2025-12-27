"""
Walkthrough conversation generation for process-oriented synthetic data.

Generates step-by-step guided conversations that help users learn how to do
something - like character creation, mech building, or session prep.
"""

import json
from typing import Dict, List, Optional

from ..utils.llm_client import call_llm
from .synth_io import log_invalid_response
from .synth_llm import WarningLimiter
from .synth_prompts import PromptConfig


def generate_walkthrough_conversation(
    text_chunk: str,
    prompt_config: PromptConfig,
    walkthrough_topic: str,
    model: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    n_turns: int = 3,
    citation_style: str = "(p. {page})",
    grounding_instructions: str = "",
    format_instructions: str = "",
    repair_invalid_json: bool = True,
    invalid_log_path: Optional[str] = None,
    warning_limiter: Optional[WarningLimiter] = None,
) -> Optional[Dict]:
    """
    Generate a guided walkthrough conversation.

    Args:
        text_chunk: Source context containing the process/steps
        prompt_config: Configured prompt templates
        walkthrough_topic: What the walkthrough is about (e.g., "character creation")
        model: LLM model to use
        n_turns: Number of exchange turns (3 = 6 messages total)

    Returns:
        Dict with 'messages' list and metadata, or None if generation fails
    """
    prompt = prompt_config.format_walkthrough_prompt(
        text=text_chunk,
        walkthrough_topic=walkthrough_topic,
        n_turns=n_turns,
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
        msg = "Warning: Empty walkthrough response from model."
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
            "topic_summary": data.get("topic_summary", f"walkthrough: {walkthrough_topic}"),
            "task_type": data.get("task_type", "character_build"),
            "turn_count": len(messages) // 2,
            "walkthrough_topic": walkthrough_topic,
            "is_walkthrough": True,
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        msg = f"Warning: Failed to parse walkthrough response: {e}"
        if warning_limiter:
            warning_limiter.warn(msg)
        else:
            print(msg)

        if invalid_log_path:
            log_invalid_response(invalid_log_path, response)

        return None


def generate_walkthrough_series(
    chunks: List[Dict],
    prompt_config: PromptConfig,
    walkthrough_topic: str,
    model: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    n_conversations: int = 5,
    turns_per_conversation: int = 3,
    citation_style: str = "(p. {page})",
    grounding_instructions: str = "",
    format_instructions: str = "",
    rag_system_prompt: str = "",
    repair_invalid_json: bool = True,
    invalid_log_path: Optional[str] = None,
    warning_limiter: Optional[WarningLimiter] = None,
) -> List[Dict]:
    """
    Generate a series of walkthrough conversations from multiple chunks.

    This combines context from multiple chunks to create comprehensive
    walkthrough conversations that span the full process.

    Args:
        chunks: Filtered chunks relevant to the walkthrough topic
        n_conversations: How many conversations to generate
        turns_per_conversation: Number of turns per conversation

    Returns:
        List of conversation records ready for training
    """
    if not chunks:
        print("Warning: No chunks provided for walkthrough generation.")
        return []

    results = []

    # Distribute conversations across chunks
    # For small chunk counts, some chunks get multiple conversations
    # For large chunk counts, we sample
    chunk_assignments = []
    if len(chunks) >= n_conversations:
        # Sample chunks evenly
        step = len(chunks) // n_conversations
        for i in range(n_conversations):
            idx = min(i * step, len(chunks) - 1)
            chunk_assignments.append(idx)
    else:
        # Use each chunk, then cycle
        for i in range(n_conversations):
            chunk_assignments.append(i % len(chunks))

    for conv_idx, chunk_idx in enumerate(chunk_assignments):
        chunk = chunks[chunk_idx]
        text = chunk.get("text_prefixed") or chunk.get("text", "")
        page = chunk.get("page_start") or chunk.get("page", "unknown")

        # Build context pages list
        context_pages = []
        if isinstance(chunk.get("page_start"), int):
            if isinstance(chunk.get("page_end"), int):
                context_pages = list(range(chunk["page_start"], chunk["page_end"] + 1))
            else:
                context_pages = [chunk["page_start"]]
        elif isinstance(page, int):
            context_pages = [page]

        conversation = generate_walkthrough_conversation(
            text_chunk=text,
            prompt_config=prompt_config,
            walkthrough_topic=walkthrough_topic,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            n_turns=turns_per_conversation,
            citation_style=citation_style,
            grounding_instructions=grounding_instructions,
            format_instructions=format_instructions,
            repair_invalid_json=repair_invalid_json,
            invalid_log_path=invalid_log_path,
            warning_limiter=warning_limiter,
        )

        if conversation:
            # Format as training record - use same schema as generate_synthetic.py
            record = {
                "instruction": conversation["messages"][0]["content"],
                "input": text,  # Match main generator's 'input' field
                "output": conversation["messages"][-1]["content"],
                "task_type": conversation["task_type"],
                "source_page": page,
                "context": text,
                "citations": context_pages,
                "answer_format": "walkthrough",  # Match main generator's answer_format
                # Walkthrough-specific metadata
                "is_multiturn": True,
                "is_walkthrough": True,
                "turn_count": conversation["turn_count"],
                "topic_summary": conversation["topic_summary"],
                "walkthrough_topic": walkthrough_topic,
            }

            # Add full messages for chat fine-tuning
            if rag_system_prompt:
                record["messages"] = [
                    {"role": "system", "content": rag_system_prompt},
                ] + conversation["messages"]
            else:
                record["messages"] = conversation["messages"]

            # Add chunk provenance - match main generator's field names
            if chunk.get("doc_id"):
                record["source_doc_id"] = chunk["doc_id"]
            if chunk.get("chunk_id"):
                record["source_chunk_id"] = chunk["chunk_id"]
            if isinstance(chunk.get("page_start"), int):
                record["source_page_start"] = chunk["page_start"]
            if isinstance(chunk.get("page_end"), int):
                record["source_page_end"] = chunk["page_end"]
            if chunk.get("section_path"):
                record["source_section"] = " > ".join(str(s) for s in chunk["section_path"])

            results.append(record)

    return results

