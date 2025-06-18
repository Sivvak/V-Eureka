import os
import base64
from typing import List, Dict, Optional
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import google.generativeai as genai

# THIS FILE IS A DEMONSTRATION HOW TO SEND IMAGES. You can test it by creating `images` directory and putting some png image here


def encode_image_to_bytes(path: str) -> bytes:
    with open(path, "rb") as img_file:
        return img_file.read()


def generate_reward_functions(
    model: str,
    messages: List[Dict[str, str]],
    chunk_size: int,
    cfg: DictConfig,
    image_filenames: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:

    prompt_parts = [m["content"] for m in messages]

    images = []
    if image_filenames:
        for filename in image_filenames:
            image_path = os.path.join("images", filename)
            uploaded_file = genai.upload_file(path=image_path)
            images.append(uploaded_file)

    parts = []
    for p in prompt_parts:
        parts.append({"text": p})
    for img in images:
        parts.append({"file_data": img})

    client = genai.GenerativeModel(model)
    choices = []
    for _ in range(chunk_size):
        response = client.generate_content(parts)
        choices.append({"message": {"content": response.text}})

    return {"choices": choices}


def run_generate_reward_functions():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

    cfg = OmegaConf.create({
        "model": "gemma-3-4b-it"
    })

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate reward functions for locomotion policy."},
        {"role": "user", "content": "Consider stability and efficiency."},
        {"role": "system", "content": "Keep explanations simple and concise."}
    ]

    image_filenames = ["f16v2.png"]  # Optional images in ./images/
    chunk_size = 1

    response = generate_reward_functions(cfg.model, messages, chunk_size, cfg, image_filenames)

    for i, choice in enumerate(response["choices"], 1):
        print(f"\nReward Function {i}:\n{choice['message']['content']}\n{'-'*40}")


if __name__ == "__main__":
    run_generate_reward_functions()