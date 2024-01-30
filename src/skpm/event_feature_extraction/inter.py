import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from skpm.config import EventLogConfig as elc


class WorkInProgress(TransformerMixin, BaseEstimator):
    """Work in progress (WIP) feature extractor.

    Args:
        window_size: {str}, default='1D'
            Frequency of the bins to count the number of cases (work) in progress. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

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
    ):
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
            pd.Grouper(key=elc.timestamp, freq=self.window_size)
        )[elc.case_id].nunique()
        self._bins = pd.cut(
            X[elc.timestamp],
            bins=self._grouped_wip.index,
            labels=self._grouped_wip.index[:-1],
        )
        wip = self._bins.map(self._grouped_wip)
        wip = wip.fillna(self._bins.isna().sum()).values

        return wip
