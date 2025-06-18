from typing import Dict, List
from omegaconf import DictConfig
import google.generativeai as genai

def generate_reward_functions(
    model: str,
    messages: List[Dict[str, str]],
    chunk_size: int,
    cfg: DictConfig,
) -> Dict[str, List[Dict[str, Dict[str, str]]]]:
    '''
    Function for generating propositions of reward functions for eureka algorithm.
    Input:
        - model: LLM model that generates the response
        - messages: input for the model. 
                    it is either a list of two prompts (initial_user prompt and initial_system prompt)
                    or a list of four messages (initial_user prompt, initial_system prompt and two reflection prompts)
        - chunk_size: number of reward functions to generate
        - cfg: additional configuration
    Output:
        - response: a dictionary of LLM resposes.
                    Key `choices` contains a list of generated reward functions.
                    You can add other keys to track statistics e.g. `number_of_tokens`.
    '''

    prompt_text = "\n".join(m["content"] for m in messages)

    gen_model = genai.GenerativeModel(model)

    choices = []
    for _ in range(chunk_size):
        response = gen_model.generate_content(prompt_text)
        choices.append({"message": {"content": response.text}})

    return {"choices": choices}