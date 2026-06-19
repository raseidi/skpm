from numbers import Integral, Real

import pandas as pd
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class Indexing(BaseProcessTransformer):
    """Position-based lag encoding per case.

    For each event, build columns ``<attr>_pos_1, <attr>_pos_2, ...,
    <attr>_pos_n`` holding the value of ``attr`` at the i-th lag within
    the case. Lags use ``groupby(level="case_id").shift(i, fill_value=...)``.
    """

    _parameter_constraints = {
        "n": [Interval(type=Integral, left=1, right=None, closed="left"), None],
        "attributes": [str, list, None],
        "fill_cat_value": [int, str, None],
        "fill_num_value": [Real, None],
    }

    def __init__(
        self,
        n: int | None = 2,
        attributes: str | list[str] | None = None,
        fill_cat_value: int | str | None = None,
        fill_num_value: float | None = None,
    ):
        self.n = n
        self.attributes = attributes
        self.fill_cat_value = fill_cat_value
        self.fill_num_value = fill_num_value

    def _fit(self, X: pd.DataFrame, y=None):
        # Resolve which attributes to lag without touching the param, so the
        # estimator survives clone/get_params and a second transform is stable.
        if self.attributes is None:
            self.attributes_ = X.columns.tolist()
        elif isinstance(self.attributes, str):
            self.attributes_ = [self.attributes]
        else:
            self.attributes_ = list(self.attributes)

        # Fix the lag set at fit so the output feature space is stable. With
        # n=None the lag count is data-dependent (longest case minus one), so
        # it must be pinned here rather than recomputed per transform.
        if self.n is not None:
            self.lags_ = list(range(1, self.n + 1))
        else:
            max_case_len = (
                X.groupby(level=self.case_id, sort=False, observed=True)
                .size()
                .max()
            )
            self.lags_ = list(range(1, max_case_len))

        self.feature_names_out_ = [
            f"{col}_pos_{lag}" for col in self.attributes_ for lag in self.lags_
        ]
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "attributes_")

        group = X.groupby(level=self.case_id, sort=False, observed=True)

        num_attributes = X.select_dtypes(include=float).columns
        time_attributes = X.select_dtypes(
            include=["datetime", "timedelta", "datetimetz"]
        ).columns

        out = pd.DataFrame(index=X.index)
        for col in self.attributes_:
            if col in num_attributes:
                fill_value = self.fill_num_value
            elif col in time_attributes:
                fill_value = None
            else:
                fill_value = self.fill_cat_value

            lagged_cols = [f"{col}_pos_{lag}" for lag in self.lags_]
            shifted = group[col].shift(self.lags_, fill_value=fill_value)
            shifted.columns = lagged_cols
            for c in lagged_cols:
                out[c] = shifted[c]

        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return list(self.feature_names_out_)
