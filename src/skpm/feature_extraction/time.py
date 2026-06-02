import inspect
from typing import Optional, Union

import pandas as pd
from sklearn.base import ClassNamePrefixFeaturesOutMixin, check_is_fitted
from sklearn.calibration import StrOptions

from skpm.base import BaseProcessTransformer
from skpm.feature_extraction.case.time import TimestampCaseLevel
from skpm.feature_extraction.event.time import TimestampEventLevel
from skpm.utils import validate_methods_from_class


def _to_list(x) -> list:
    if x == "all" or x is None:
        return x
    return [x] if not isinstance(x, list) else x


class TimestampExtractor(ClassNamePrefixFeaturesOutMixin, BaseProcessTransformer):
    """Extract timestamp-derived features and targets from an event log.

    Input is an event-log DataFrame carrying the canonical
    ``MultiIndex(case_id, timestamp, event_id)``. Output is a DataFrame
    with the same index and one column per requested feature/target.
    The transformer does not return ``case_id`` or ``timestamp`` in the
    output columns — those live in the index and travel through the
    Pipeline implicitly.

    Targets: ``execution_time``, ``remaining_time``.
    """

    available_targets = ["execution_time", "remaining_time"]

    _parameter_constraints = {
        "case_features": [StrOptions({"all"}), list, None],
        "event_features": [StrOptions({"all"}), list, None],
        "targets": [StrOptions({"remaining_time"}), list, None],
        "time_unit": [StrOptions({"s", "m", "h", "d"})],
    }

    def __init__(
        self,
        case_features: Union[str, list, None] = "all",
        event_features: Union[str, list, None] = "all",
        targets: Optional[Union[str, list]] = None,
        time_unit: str = "s",
    ):
        self.case_features = _to_list(case_features)
        self.event_features = _to_list(event_features)
        self.targets = _to_list(targets)
        self.time_unit = time_unit

    def _fit(self, X: pd.DataFrame, y=None):
        self.event_features = validate_methods_from_class(
            class_obj=TimestampEventLevel, methods=self.event_features
        )
        self.case_features = validate_methods_from_class(
            class_obj=TimestampCaseLevel, methods=self.case_features
        )
        self.targets = validate_methods_from_class(
            class_obj=TimestampCaseLevel, methods=self.targets
        )

        self._n_features_out = len(self.event_features) + len(self.case_features)
        self._n_targets_out = len(self.targets)

        if self._n_features_out + self._n_targets_out == 0:
            raise ValueError(
                "No features selected. Please select at least one feature, "
                "either from the event level or the case level."
            )

        return self

    def get_feature_names_out(self):
        return [
            f[0] for f in self.case_features + self.event_features + self.targets
        ]

    def _transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, "_n_features_out")

        timestamps = self._timestamps(X)
        out = pd.DataFrame(index=X.index)

        for feature_name, feature_fn in (
            self.case_features + self.event_features + self.targets
        ):
            kwargs = (
                {"time_unit": self.time_unit}
                if "time_unit" in inspect.signature(feature_fn).parameters
                else {}
            )
            out[feature_name] = feature_fn(timestamps, **kwargs)

        return out
