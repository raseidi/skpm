from skpm.config import EventLogConfig as elc
from sklearn.base import TransformerMixin, _fit_context
from skpm.base import BaseProcessEstimator
from sklearn.utils._param_validation import Interval, _IterablesNotString, Options
from numbers import Integral, Real
import numpy as np
import pandas as pd
from typing import Union

class Indexing(TransformerMixin, BaseProcessEstimator):
    _parameter_constraints = {
            "n": [Interval(type=Integral, left=1, right=None, closed="left"), None],
            "attributes": [str, list, None],
            "fill_value": [Real, None],
    }
    def __init__(self, n: int = 2, attributes: Union[str, list] = None, fill_value: int = None):
        self.n = n
        self.attributes = attributes
        self.fill_value = fill_value

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(self.attributes, str):
            self.attributes = [self.attributes]
        
        if self.attributes is None:
            self.attributes = X.columns.difference([elc.case_id]).tolist()
        return self

    def transform(self, X: pd.DataFrame, y=None):
        group = X.groupby(elc.case_id)
        
        out_df = pd.DataFrame()
        lags = range(1, self.n + 1)
        for col in self.attributes:
            lagged_cols = [f"{col}_lag_{lag}" for lag in lags]
            out_df[lagged_cols] = group[col].shift(lags, fill_value=self.fill_value)
                
        return out_df