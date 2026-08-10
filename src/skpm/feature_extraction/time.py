import inspect
from typing import Union

import pandas as pd
from sklearn.base import ClassNamePrefixFeaturesOutMixin, check_is_fitted
from sklearn.calibration import StrOptions

from skpm.base import BaseProcessTransformer
from skpm.feature_extraction.case.time import TimestampCaseLevel
from skpm.feature_extraction.event.time import TimestampEventLevel
from skpm.utils import resolve_features


class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, BaseProcessTransformer
):
    """Extract timestamp-derived features from an event log.

    Input is an event-log DataFrame carrying the canonical
    ``MultiIndex(case_id, timestamp, event_id)``. Output is a DataFrame
    with the same index and one column per requested feature.
    The transformer does not return ``case_id`` or ``timestamp`` in the
    output columns — those live in the index and travel through the
    Pipeline implicitly.

    Only past-looking features are exposed: values derived from future
    events (e.g. remaining time until case end) would leak the prediction
    target into the feature matrix. Build labels with
    ``skpm.feature_extraction.targets`` and pass them as ``y`` instead.
    """

    # A bare string selects a single feature (e.g. ``"accumulated_time"``) or
    # the ``"all"`` shortcut; a list selects several; ``None`` selects none.
    # Names are validated against each level's ``FEATURES`` registry in
    # ``_fit``.
    _parameter_constraints = {
        "case_features": [str, list, None],
        "event_features": [str, list, None],
        "time_unit": [StrOptions({"s", "m", "h", "d", "w"})],
    }

    def __init__(
        self,
        case_features: Union[str, list, None] = "all",
        event_features: Union[str, list, None] = "all",
        time_unit: str = "s",
    ):
        self.case_features = case_features
        self.event_features = event_features
        self.time_unit = time_unit

    def _fit(self, X: pd.DataFrame, y=None):
        self.event_features_ = resolve_features(
            class_obj=TimestampEventLevel, selection=self.event_features
        )
        self.case_features_ = resolve_features(
            class_obj=TimestampCaseLevel, selection=self.case_features
        )

        self._n_features_out = len(self.event_features_) + len(
            self.case_features_
        )

        if self._n_features_out == 0:
            raise ValueError(
                "No features selected. Please select at least one feature, "
                "either from the event level or the case level."
            )

        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_n_features_out")
        return [f[0] for f in self.case_features_ + self.event_features_]

    def _transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, "_n_features_out")

        timestamps = self._timestamps(X)
        out = pd.DataFrame(index=X.index)

        for feature_name, feature_fn in (
            self.case_features_ + self.event_features_
        ):
            kwargs = (
                {"time_unit": self.time_unit}
                if "time_unit" in inspect.signature(feature_fn).parameters
                else {}
            )
            out[feature_name] = feature_fn(timestamps, **kwargs)

        return out
