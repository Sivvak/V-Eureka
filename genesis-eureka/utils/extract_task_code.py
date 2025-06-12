import ast


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = (
        function_def.name
        + "(self."
        + ", self.".join(arg.arg for arg in function_def.args.args)
        + ")"
    )
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst
