def file_to_string(filename):
    """Read a file and return its contents as a string."""
    with open(filename, "r") as file:
        return file.read()
