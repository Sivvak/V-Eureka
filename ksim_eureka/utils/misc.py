import logging
import os
import subprocess
import time


def set_freest_gpu():
    """Set the CUDA_VISIBLE_DEVICES to the GPU with the most free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            free_memory = [int(x) for x in result.stdout.strip().split("\n")]
            freest_gpu = free_memory.index(max(free_memory))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(freest_gpu)
            logging.info(
                f"Set CUDA_VISIBLE_DEVICES to GPU {freest_gpu} with {max(free_memory)} MB free memory"
            )
        else:
            logging.warning("Could not query GPU memory, using default GPU")
    except Exception as e:
        logging.warning(f"Error setting GPU: {e}, using default GPU")


def block_until_training(log_file, log_status=False, iter_num=None, response_id=None):
    """Block until training is complete by monitoring the log file."""
    logging.info(f"Waiting for training to complete: {log_file}")

    while True:
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    content = f.read()

                # Check for completion indicators in K-Sim logs
                if "Exiting training job" in content:
                    if log_status and iter_num is not None and response_id is not None:
                        logging.info(
                            f"Iteration {iter_num}, Response {response_id}: Exiting training job"
                        )
                    break

                # Check for error conditions
                if "Error" in content or "Exception" in content:
                    if log_status and iter_num is not None and response_id is not None:
                        logging.warning(
                            f"Iteration {iter_num}, Response {response_id}: Training error detected"
                        )
                    break

            except Exception as e:
                logging.warning(f"Error reading log file {log_file}: {e}")

        time.sleep(5)  # Check every 5 seconds
