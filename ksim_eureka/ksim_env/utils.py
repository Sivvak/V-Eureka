import logging


def save_reward_class(reward_code: str, output_file: str):
    """Save the generated reward class to a file."""
    try:
        with open(output_file, "w") as f:
            f.write(reward_code)
        logging.info(f"Saved reward class to {output_file}")
    except Exception as e:
        logging.error(f"Error saving reward class to {output_file}: {e}")
