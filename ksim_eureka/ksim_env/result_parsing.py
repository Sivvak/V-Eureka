def filter_traceback(stdout_str):
    """Filter and extract relevant error information from stdout."""

    # TODO: possibly needs adjustments

    lines = stdout_str.split("\n")
    error_lines = []

    for line in lines:
        if any(
            keyword in line.lower()
            for keyword in ["error", "exception", "traceback", "failed"]
        ):
            error_lines.append(line.strip())

    if error_lines:
        return "\n".join(error_lines[-10:])  # Return last 10 error lines
    return ""
