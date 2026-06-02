from numbers import Integral, Real

import pandas as pd
from sklearn.utils._param_validation import Interval

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

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self._validate_log(X, copy=False)

        if self.attributes is None:
            self.attributes = X.columns.tolist()
        elif isinstance(self.attributes, str):
            self.attributes = [self.attributes]

        group = X.groupby(level="case_id", sort=False, observed=True)
        lags = range(1, self.n + 1) if self.n is not None else range(1, group.size().max())

        num_attributes = X.select_dtypes(include=float).columns
        time_attributes = X.select_dtypes(
            include=["datetime", "timedelta", "datetimetz"]
        ).columns

        out = pd.DataFrame(index=X.index)
        for col in self.attributes:
            if col in num_attributes:
                fill_value = self.fill_num_value
            elif col in time_attributes:
                fill_value = None
            else:
                fill_value = self.fill_cat_value

            lagged_cols = [f"{col}_pos_{lag}" for lag in lags]
            shifted = group[col].shift(lags, fill_value=fill_value)
            shifted.columns = lagged_cols
            for c in lagged_cols:
                out[c] = shifted[c]

        return out
