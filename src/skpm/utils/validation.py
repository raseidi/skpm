from typing import Any, Callable, Iterable, Union


def resolve_features(
    class_obj: type, selection: Union[str, list[str], None]
) -> list[tuple[str, Callable]]:
    """Resolve a feature selection against a feature class registry.

    ``class_obj`` must declare a ``FEATURES`` tuple naming the classmethods
    that may be used as input features. An optional ``TARGET_ONLY`` tuple
    names computations reserved for label generation (they use future
    information and would leak the prediction target if used as features).

    Args:
        class_obj (type): class declaring ``FEATURES`` (and optionally
            ``TARGET_ONLY``).
        selection (Union[str, list[str], None]): ``None`` selects nothing,
            ``"all"`` selects every registered feature, a string selects a
            single feature, a list selects several (deduplicated, user order).

    Returns:
        list[tuple[str, Callable]]: (name, bound classmethod) pairs.

    Raises:
        TypeError: if ``class_obj`` does not declare ``FEATURES``.
        ValueError: for names in ``TARGET_ONLY`` or names not registered.
    """
    features = getattr(class_obj, "FEATURES", None)
    if features is None:
        raise TypeError(
            f"{class_obj.__name__} does not declare a FEATURES registry."
        )

    if selection is None:
        return []
    if isinstance(selection, str) and selection == "all":
        names = list(features)
    else:
        if isinstance(selection, str):
            selection = [selection]
        names = list(dict.fromkeys(selection))

    target_only = getattr(class_obj, "TARGET_ONLY", ())
    for name in names:
        if name in target_only:
            raise ValueError(
                f"'{name}' is a prediction target, not a feature: using it "
                "as an input feature leaks future information. Build the "
                f"label with skpm.feature_extraction.targets.{name}(log) "
                "and pass it as y instead."
            )
    unknown = [name for name in names if name not in features]
    if unknown:
        raise ValueError(
            f"Unknown feature(s) {unknown} for {class_obj.__name__}. "
            f"Available features: {list(features)}."
        )

    return [(name, getattr(class_obj, name)) for name in names]


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
