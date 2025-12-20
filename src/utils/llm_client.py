import os
from typing import Any, List, Optional

from openai import OpenAI

SYSTEM_PROMPT = "You are a helpful assistant for generating synthetic RPG data."


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""

    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text
    return ""


def _build_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _call_chat(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: Optional[float],
    max_completion_tokens: Optional[int],
    max_tokens: Optional[int],
) -> str:
    request_kwargs = {
        "model": model,
        "messages": _build_messages(prompt),
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_completion_tokens
    elif max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        message = str(e)
        retry_kwargs = dict(request_kwargs)

        if "temperature" in message and "unsupported" in message:
            retry_kwargs.pop("temperature", None)
        if "max_completion_tokens" in message and "unsupported" in message:
            retry_kwargs.pop("max_completion_tokens", None)
            if max_tokens is not None:
                retry_kwargs["max_tokens"] = max_tokens
        if "max_tokens" in message and "unsupported" in message:
            retry_kwargs.pop("max_tokens", None)
            if max_completion_tokens is not None:
                retry_kwargs["max_completion_tokens"] = max_completion_tokens

        if retry_kwargs != request_kwargs:
            response = client.chat.completions.create(**retry_kwargs)
            return response.choices[0].message.content or ""

        raise


def _call_responses(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    max_tokens: Optional[int],
) -> str:
    request_kwargs = {
        "model": model,
        "input": prompt,
        "instructions": SYSTEM_PROMPT,
        "text": {"format": {"type": "text"}},
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = max_output_tokens
    elif max_completion_tokens is not None:
        request_kwargs["max_output_tokens"] = max_completion_tokens
    elif max_tokens is not None:
        request_kwargs["max_output_tokens"] = max_tokens

    try:
        response = client.responses.create(**request_kwargs)
        status = getattr(response, "status", None)
        if status and status != "completed":
            details = getattr(response, "incomplete_details", None)
            print(f"Warning: Response incomplete (status={status}, details={details}).")
        text = _extract_response_text(response)
        if not text.strip():
            print("Warning: Empty response text from Responses API.")
        return text
    except Exception as e:
        message = str(e)
        retry_kwargs = dict(request_kwargs)

        if "temperature" in message and "unsupported" in message:
            retry_kwargs.pop("temperature", None)
        if "max_output_tokens" in message and "unsupported" in message:
            retry_kwargs.pop("max_output_tokens", None)

        if retry_kwargs != request_kwargs:
            response = client.responses.create(**retry_kwargs)
            return _extract_response_text(response)

        raise

def call_llm(
    prompt: str,
    model: str = "gpt-4o",
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Calls an LLM (OpenAI compatible) to generate a response.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment. Returning mock response.")
        # Return a valid JSON list structure for the smoke test to parse
        return """
[
  {
    "instruction": "Explain the basic combat mechanic.",
    "output": "Combat is turn-based, involving move and action phases.",
    "thought_process": "Simulated reasoning for smoke test."
  },
  {
    "instruction": "What is a mech?",
    "output": "A mech is a giant robot piloted by a player character.",
    "thought_process": "Checking definitions in mock context."
  }
]
"""

    try:
        client = OpenAI(api_key=api_key)

        if model.startswith("gpt-5"):
            return _call_responses(
                client,
                model,
                prompt,
                temperature,
                max_output_tokens,
                max_completion_tokens,
                max_tokens,
            )

        content = _call_chat(client, model, prompt, temperature, max_completion_tokens, max_tokens)
        if content.strip():
            return content

        return _call_responses(
            client,
            model,
            prompt,
            temperature,
            max_output_tokens,
            max_completion_tokens,
            max_tokens,
        )
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""
