import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from ksim_env.code_enrichment import add_required_imports
from ksim_env.code_extraction import (
    extract_reward_class_from_code,
    validate_reward_class_code,
)
from ksim_env.code_generation import create_task_with_reward
from ksim_env.env_information import get_available_observations
from ksim_env.result_parsing import filter_traceback
from ksim_env.utils import save_reward_class
from mocks.response import get_mock_response
from utils.file_utils import file_to_string
from utils.misc import block_until_training, set_freest_gpu

KSIM_EUREKA_ROOT_DIR = os.getcwd()


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {KSIM_EUREKA_ROOT_DIR}")

    # START TODO: Configure LLM
    ...
    # END TODO

    task_name = cfg.task.name
    task_description = cfg.task.description
    base_task_file = cfg.task.base_task_file
    model = cfg.model

    logging.info(f"Using LLM: {model}")
    logging.info(f"Task: {task_name}")
    logging.info(f"Task description: {task_description}")
    logging.info(f"Base task file: {base_task_file}")

    # Load prompts
    prompt_dir = f"{KSIM_EUREKA_ROOT_DIR}/prompts"
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )

    # Format prompts
    initial_system = (
        initial_system.format(task_reward_signature_string=reward_signature)
        + code_output_tip
    )
    available_observations = get_available_observations()
    initial_user = initial_user.format(
        task_description=task_description, available_observations=available_observations
    )

    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]

    # Initialize tracking variables
    DUMMY_FAILURE = -10000.0
    max_successes = []
    max_success_rewards = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_overall = DUMMY_FAILURE
    max_reward_code_path = None

    # Main Eureka generation loop
    for iter in range(cfg.iteration):
        logging.info(
            f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}"
        )

        # Generate responses from LLM
        responses = []
        total_samples = 0
        total_tokens = 0
        total_completion_tokens = 0
        chunk_size = 4

        while total_samples < cfg.sample:
            current_batch_size = min(chunk_size, cfg.sample - total_samples)

            for sample_idx in range(current_batch_size):
                for attempt in range(1000):
                    try:
                        # START TODO: Switch to actually generating response
                        ...
                        # END TODO

                        gemini_response = get_mock_response(
                            messages=messages,
                            temperature=cfg.temperature,
                            iteration=iter,
                            response_id=total_samples,
                        )

                        responses.append(gemini_response)
                        total_samples += 1

                        # START TODO: Retrieve real usage
                        ...
                        # END TODO

                        # Mock values
                        prompt_tokens = 500
                        total_completion_tokens += 300
                        total_tokens += 800

                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                            logging.info(f"Reduced chunk size to {chunk_size}")
                        logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                        time.sleep(1)

                if sample_idx >= current_batch_size - 1:
                    break

            if len(responses) == 0:
                logging.error("Code terminated due to too many failed attempts!")
                exit()

        if cfg.sample == 1:
            logging.info(
                f"Iteration {iter}: Gemini Output:\n{responses[0]['content']}\n"
            )

        logging.info(
            f"Iteration {iter}: Estimated Prompt Tokens: {prompt_tokens:.0f}, "
            f"Completion Tokens: {total_completion_tokens:.0f}, Total Tokens: {total_tokens:.0f}"
        )

        # Process each generated reward class
        code_runs = []
        rl_runs = []

        for response_id in range(len(responses)):
            response_content = responses[response_id]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            try:
                reward_class_name, reward_code = extract_reward_class_from_code(
                    response_content
                )

                if not validate_reward_class_code(reward_code):
                    logging.warning(
                        f"Invalid reward class generated for response {response_id}"
                    )
                    continue

                reward_code = add_required_imports(reward_code)

                reward_file = f"reward_iter{iter}_response{response_id}.py"
                save_reward_class(reward_code, reward_file)
                code_runs.append(reward_code)

                task_file = f"task_iter{iter}_response{response_id}.py"
                create_task_with_reward(
                    base_task_file=base_task_file,
                    reward_class_code=reward_code,
                    output_file=task_file,
                    reward_class_name=reward_class_name,
                )

                tasks_dir = os.path.join(KSIM_EUREKA_ROOT_DIR, "tasks")
                shutil.copytree(
                    tasks_dir,
                    os.path.join(workspace_dir, "tasks"),
                    dirs_exist_ok=True,
                )

                shutil.copy(task_file, os.path.join(workspace_dir, "tasks", task_file))

                # Set GPU for training
                set_freest_gpu()

                # Execute K-Sim training
                log_file = f"training_iter{iter}_response{response_id}.txt"

                # Construct training command for K-Sim
                training_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    f"tasks.{task_file.replace('.py', '')}",
                    f"max_steps={cfg.max_iterations}",
                    f"num_envs={cfg.get('num_envs', cfg.num_envs)}",
                    f"batch_size={cfg.get('batch_size', cfg.batch_size)}",
                ]

                with open(log_file, "w") as f:
                    process = subprocess.Popen(
                        training_cmd, stdout=f, stderr=f, cwd=workspace_dir
                    )

                # Wait for training to complete
                block_until_training(
                    log_file, log_status=True, iter_num=iter, response_id=response_id
                )
                rl_runs.append(process)

            except Exception as e:
                logging.error(f"Error processing response {response_id}: {e}")
                continue

        # Analyze training results
        code_feedbacks = []
        contents = []
        successes = []
        success_rewards = []
        code_paths = []

        exec_success = False

        for response_id, rl_run in enumerate(rl_runs):
            rl_run.communicate()
            log_file = f"training_iter{iter}_response{response_id}.txt"
            code_paths.append(f"reward_iter{iter}_response{response_id}.py")

            try:
                with open(log_file, "r") as f:
                    stdout_str = f.read()
            except:
                content = execution_error_feedback.format(
                    traceback_msg="Training execution failed! Please check the reward class implementation."
                )
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                success_rewards.append(DUMMY_FAILURE)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                # START TODO: Gather RL training results and construct reward reflection
                ...
                # END TODO

            else:
                # Training failed
                successes.append(DUMMY_FAILURE)
                success_rewards.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # Check if any training succeeded
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            max_success_rewards.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeating iteration...")
            continue

        # Select best performing reward
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward = success_rewards[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / len(successes)

        # Update overall best
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_overall = max_success_reward
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_success_rewards.append(max_success_reward)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(
            f"Iteration {iter}: Max Success: {max_success:.4f}, "
            f"Execute Rate: {execute_rate:.4f}, Max Success Reward: {max_success_reward:.4f}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(
            f"Iteration {iter}: Gemini Output Content:\n{responses[best_sample_idx]['content']}\n"
        )
        logging.info(f"Iteration {iter}: User Content:\n{best_content}\n")

        # Plot progress
        fig, axs = plt.subplots(2, figsize=(10, 8))
        fig.suptitle(f"K-Sim Eureka Progress: {task_name}")

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success Rate")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Success Rate")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Execute Rate")

        plt.tight_layout()
        plt.savefig("ksim_eureka_progress.png")

        # Save progress data
        np.savez(
            "ksim_eureka_progress.npz",
            max_successes=max_successes,
            execute_rates=execute_rates,
            best_code_paths=best_code_paths,
            max_success_rewards=max_success_rewards,
        )

        # Update conversation
        if len(messages) == 2:
            messages += [
                {
                    "role": "assistant",
                    "content": responses[best_sample_idx]["content"],
                },
                {"role": "user", "content": best_content},
            ]
        else:
            assert len(messages) == 4
            messages[-2] = {
                "role": "assistant",
                "content": responses[best_sample_idx]["content"],
            }
            messages[-1] = {"role": "user", "content": best_content}

        # Save conversation
        with open("ksim_eureka_messages.json", "w") as file:
            json.dump(messages, file, indent=4)

    # Final evaluation
    if max_reward_code_path is None:
        logging.error(
            "All iterations failed! Please check the task configuration and base task file."
        )
        return

    logging.info(
        f"Task: {task_name}, Max Success: {max_success_overall:.4f}, "
        f"Max Reward: {max_success_reward_overall:.4f}, Best Code: {max_reward_code_path}"
    )

    # START TODO: Evaluate the best reward code many times
    ...
    # END TODO

    # Copy best reward for final evaluation
    shutil.copy(max_reward_code_path, "best_reward_class.py")

    logging.info("K-Sim Eureka completed successfully!")
    logging.info("Best reward class saved as: best_reward_class.py")


if __name__ == "__main__":
    main()
