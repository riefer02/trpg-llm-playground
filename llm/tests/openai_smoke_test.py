import argparse
import os
from typing import Optional

from openai import OpenAI
import yaml


def extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return part_text
    return ""


def load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_model(config_path: str, model_override: Optional[str]) -> str:
    if model_override:
        return model_override
    if not os.path.exists(config_path):
        return "gpt-5-mini"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    llm_config = config.get("llm", {}) or {}
    return llm_config.get("model", "gpt-5-mini")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI connectivity test.")
    parser.add_argument("--env", default=".env", help="Path to env file with OPENAI_API_KEY")
    parser.add_argument("--config", default="config/synthetic_generic.yaml", help="Config file for model")
    parser.add_argument("--model", default=None, help="Override model name")
    args = parser.parse_args()

    load_env_file(args.env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment or env file.")
        return

    model = resolve_model(args.config, args.model)
    print(f"Using model: {model}")

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input="Reply with 'ok'.",
            instructions="You are a helpful assistant.",
            text={"format": {"type": "text"}},
        )
        content = extract_response_text(response)
        print("Success. Response:")
        print(repr(content))
        if not content.strip():
            print("Empty content detected. Raw message:")
            print(response)
        else:
            print(f"Status: {response.status}")
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
