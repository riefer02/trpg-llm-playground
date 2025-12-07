import os
import time
from openai import OpenAI

def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    """
    Calls an LLM (OpenAI compatible) to generate a response.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment. Returning mock response.")
        return f"Mock response generated for prompt: {prompt[:30]}..."

    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating synthetic RPG data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""
