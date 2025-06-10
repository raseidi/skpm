import inspect
from typing import Iterable, Union, Any


def validate_methods_from_class(
    class_obj: object, methods: Union[str, list[str]] = "all"
) -> list[tuple[str, callable]]:
    """Validate methods from a class.

    Args:
        class_obj (Any): a class object cotaining class methods.
        methods (Union[str, list[str]]), {"all", str, list[str]}: a list of methods to validate.

    Returns:
        list[tuple[str, callable]]: a list of tuples
            containing the name of the methods and the callable.
    """
    available_methods = inspect.getmembers(
        class_obj, predicate=inspect.ismethod
    )
    out_methods = []
    if methods == "all":
        out_methods = available_methods
    else:
        if not isinstance(methods, (tuple, list)):
            methods = [methods]
        for f in available_methods:
            if f[0] in methods and not f[0].startswith("_"):
                out_methods.append(f)

    return out_methods


def validate_columns(input_columns: Iterable, required: list) -> list:
    """Validate required columns.

    This method checks if the input columns
    contain the required columns.

    Args:
        input_columns (Iterable): Input columns.
        required (list): Required columns.

    Raises:
        ValueError: If the input is missing any
        of the required columns.

    Returns:
        list: the input columns
    """
    diff = set(required) - set(input_columns)
    if diff:
        raise ValueError(f"Input is missing the following columns: {diff}.")
    return required


def ensure_list(input: Any) -> list:
    """Ensure input is a list.

    Args:
        input (Any): Input to be converted to a list.

    Returns:
        list: Input as a list.
    """
    if not isinstance(input, list):
        if isinstance(input, (str, int)):
            input = [input]
        else:
            input = list(input)
    return input
