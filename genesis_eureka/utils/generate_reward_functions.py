from typing import Dict, List
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def generate_reward_functions(
    model: str,
    messages: List[Dict[str, str]],
    chunk_size: int,
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    '''
    Function for generating propositions of reward functions for eureka algorithm.
    Input:
        - model: LLM model that generates the response
        - messages: input for the model. 
                    it is either a list of two prompts (initial_user prompt and initial_system prompt)
                    or a list of four messages (initial_user prompt, initial_system prompt and two reflection prompts)
        - chunk_size: number of reward functions to generate
    Output:
        - response: a dictionary of LLM resposes.
                    Key `choices` contains a list of generated reward functions.
                    You can add other keys to track statistics e.g. `number_of_tokens`.
    '''
    TPM = 15_000

    choices = []
    gen_model = genai.GenerativeModel(model)

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