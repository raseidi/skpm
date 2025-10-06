from numbers import Integral, Real

import pandas as pd
from sklearn.utils._param_validation import Interval

from skpm.base import BaseProcessTransformer
from skpm.config import EventLogConfig as elc


class Indexing(BaseProcessTransformer):
    _parameter_constraints = {
        "n": [Interval(type=Integral, left=1, right=None, closed="left"), None],  # type: ignore
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
        if self.attributes is None:
            self.attributes = X.columns.difference([elc.case_id]).tolist()
        elif isinstance(self.attributes, str):
            self.attributes = [self.attributes]
        
        group = X.groupby(self._case_id)

        out_df = pd.DataFrame()
        if self.n is not None:
            lags = range(1, self.n + 1)
        else:
            lags = range(1, group.size().max())

        num_attributes = X.select_dtypes(include=float)
        time_attributes = X.select_dtypes(include=['datetime', 'timedelta', 'datetimetz'])
        cat_attributes = X.select_dtypes(exclude=[float, 'datetime', 'timedelta', 'datetimetz'])

        for col in self.attributes:
            if col in num_attributes:
                fill_value = self.fill_num_value
            elif col in cat_attributes:
                fill_value = self.fill_cat_value
            elif col in time_attributes:
                fill_value = None
                
            lagged_cols = [f"{col}_pos_{lag}" for lag in lags]
            out_df[lagged_cols] = group[col].shift(
                lags, fill_value=fill_value
            )

        return out_df
