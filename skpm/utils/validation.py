import inspect
from typing import Union, Any, List, Tuple, Callable


def check_features(
    features: Union[str, list[str]], class_obj: Any
) -> list[tuple[str, callable]]:
    available_features = inspect.getmembers(class_obj, predicate=inspect.ismethod)
    out_features = []
    if features == "all":
        out_features = available_features
    else:
        if not isinstance(features, (tuple, list)):
            features = [features]
        for f in available_features:
            if f[0] in features:
                out_features.append(f)

    return out_features
