import inspect
from typing import Iterable, Union, Any


def validate_features_from_class(
    features: Union[str, list[str]], class_obj: Any
) -> list[tuple[str, callable]]:
    """Validate features.

    Args:
        features (Union[str, list[str]]): features to be extracted from a class.
        class_obj (Any): a class object cotaining class methods.

    Returns:
        list[tuple[str, callable]]: a list of tuples
            containing the name of the feature and the method.
    """
    available_features = inspect.getmembers(class_obj, predicate=inspect.ismethod)
    out_features = []
    if features == "all":
        out_features = available_features
    else:
        if not isinstance(features, (tuple, list)):
            features = [features]
        for f in available_features:
            if f[0] in features and not f[0].startswith("_"):
                out_features.append(f)

    return out_features


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
    if input is None:
        input = []
    if not isinstance(input, list):
        if isinstance(input, (str, int)):
            input = [input]
        else:
            input = list(input)
    return input
