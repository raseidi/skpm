# TODO:
# 1. work in progress (WIP)
# 2. resource occupation (RO)

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)

# from skpm.utils.config import COLUMNS as c
from skpm.utils.validation import validate_columns


class WorkInProgress(TransformerMixin, BaseEstimator):
    """Work in progress (WIP) feature extractor.

    The class needs a column with case ids and a column with timestamps. The validation of the columns is done in the fit method.

    Args:
        window_size: {str}, default='1D'
            Frequency of the bins to count the number of cases (work) in progress. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        case_col: {str}, default='case_id'
            Name of the column containing the case ids.
        time_col: {str}, default='timestamp'
            Name of the column containing the timestamps.

    Returns:
        ndarray: WIP feature array of shape (n_samples, 1)
    """

    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    def __init__(
        self,
        window_size="1D",
    ) -> None:
        self.window_size = window_size

    def get_feature_names_out(self):
        return ["wip"]

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        case_col: str = "case:concept:name",
        time_col: str = "case:time:timestamp",
    ):
        self._case_col = case_col
        self._time_col = time_col
        assert isinstance(X, pd.DataFrame), "Input must be a dataframe."

        return self

    def transform(self, X: pd.DataFrame):
        """
        1. _grouped_wip counts the number of cases within each bin
        2. pd.cut creates a new dataframe to identify to which bin each event timestamp belongs
        3. from the bins dataframe we can map the _grouped_wip to get the number of active cases at each bin (time step)
        4. fill na since the pd.cut does not consider the last bin (it works with open interval, i.e. `[)`)
        """
        self._grouped_wip = X.groupby(
            pd.Grouper(key=self._time_col, freq=self.window_size)
        )[self._case_col].nunique()
        self._bins = pd.cut(
            X[self._time_col],
            bins=self._grouped_wip.index,
            labels=self._grouped_wip.index[:-1],
        )
        wip = self._bins.map(self._grouped_wip)
        wip = wip.fillna(self._bins.isna().sum()).values

        return wip

    def get_bins(self):
        return self._bins
