from typing import Dict, List

from google import generativeai as genai
from omegaconf import DictConfig


def generate_response(model: str, messages: List[Dict[str, str]], cfg: DictConfig):
    gemini_model = genai.GenerativeModel(model)

    prompt_parts = []
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"Assistant: {msg['content']}")

    full_prompt = "\n\n".join(prompt_parts)

    generated_content = gemini_model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=cfg.temperature,
            max_output_tokens=4096,
        ),
    )

    response = {
        "content": generated_content.text,
        "finish_reason": "stop",
    }

    return response
