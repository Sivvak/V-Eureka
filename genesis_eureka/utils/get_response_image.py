import os
from typing import List, Dict
from dotenv import load_dotenv
from omegaconf import OmegaConf
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import re

""" This file demonstrates how to to run the pipeline with images. For this to work, one should provide the correct directory path where images are stored. Currently, it assumes files have names photo_<step_number>.png and adds all of them to the message with information on step number. If we wanted trajectories from many directories in one prompt, the function would have to be modified """

EXTENSION = '.png'

def collect_image_paths(img_dir):
    """collects image paths with step counts from filenames"""
    image_info = []  # will store tuples: (path, step_count)
    if not os.path.exists(img_dir):
        print(f"Directory '{img_dir}' not found.")
        return image_info
    

    pattern = re.compile(r'photo_(\d+).png$')
    
    for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext == EXTENSION:
                match = pattern.match(filename)
                step_count = int(match.group(1)) if match else 0
                image_info.append((filepath, step_count))
    
    image_info.sort(key=lambda x: x[1])
    return image_info

def upload_images_to_genai(image_info: list) -> list:
    """uploads images to genai"""
    uploaded_files = []
    for path, step_count in image_info:
        try:
            uploaded_file = genai.upload_file(path=path)
            uploaded_files.append({
                "file": uploaded_file,
                "step": step_count
            })
        except Exception as e:
            print(f"Error uploading {path}: {str(e)}")
    return uploaded_files

def generate_reward_functions(
    model: str,
    messages: List[Dict[str, str]],
    chunk_size: int,
    image_dir: str = None
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    '''
    Function for generating propositions of reward functions for eureka algorithm.
    Input:
        - model: LLM model that generates the response
        - messages: input for the model. 
                    it is either a list of two prompts (initial_user prompt and initial_system prompt)
                    or a list of four messages (initial_user prompt, initial_system prompt and two reflection prompts)
        - chunk_size: number of reward functions to generate
        - image_dir: image directory to load the files from
    Output:
        - response: a dictionary of LLM resposes.
                    Key `choices` contains a list of generated reward functions.
                    You can add other keys to track statistics e.g. `number_of_tokens`.
    '''
    TPM = 15_000

    choices = []
    gen_model = genai.GenerativeModel(model)

    if image_dir != None:
        image_info = collect_image_paths(image_dir)
        genai_images = upload_images_to_genai(image_info)

        for img_info in genai_images:
            step = img_info["step"]
            img_file = img_info["file"]
            
            messages.append({
                "role": "user",
                "parts": [
                    f"This shows the environment at step {step}",
                    img_file
                ]
            })

    tokens_count = gen_model.count_tokens(messages).total_tokens
    
    def generate_single_response():
        try:
            response = gen_model.generate_content(messages)
            return {"message": {"parts": response.text}}
        except Exception as e:
            if e.code == 429:
                # wait 60+ seconds with some jitter to avoid thundering herd
                wait_time = 60 + random.uniform(5, 15)
                time.sleep(wait_time)

                response = gen_model.generate_content(messages)
                return {"message": {"parts": response.text}}
            else:
                raise e

    with ThreadPoolExecutor(max_workers=min(TPM // tokens_count, chunk_size)) as executor:
        future_to_response = {
            executor.submit(generate_single_response): i for i in range(chunk_size)
        }

        for future in as_completed(future_to_response):
            result = future.result()
            choices.append(result)

    return {"choices": choices}

#example usage
def run_generate_reward_functions():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

    cfg = OmegaConf.create({
        "model": "gemini-2.0-flash"
    })

    messages = [
        {"role": "user", "parts": "You are a helpful assistant."},
        {"role": "user", "parts": "Generate reward functions for locomotion policy."},
        {"role": "user", "parts": "Consider stability and efficiency."},
        {"role": "user", "parts": "You should state what is the difference between the two images."}
    ]

    chunk_size = 1

    response = generate_reward_functions(cfg.model, messages, chunk_size, "images")

    for i, choice in enumerate(response["choices"], 1):
        print(f"\nReward Function {i}:\n{choice['message']['parts']}\n{'-'*40}")


if __name__ == "__main__":
    run_generate_reward_functions()