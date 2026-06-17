import pandas as pd
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class WorkInProgress(BaseProcessTransformer):
    """Work in Progress (WIP) feature extractor.

    Counts how many cases are active within each time window. Input is
    an event-log DataFrame with the canonical event-log MultiIndex
    (``case_id``, ``timestamp``, ``event_id``).

    Parameters
    ----------
    window_size : str, default='1D'
        Pandas offset alias describing the rolling time window
        (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

    Returns
    -------
    ndarray of shape (n_samples,)
    """

    _parameter_constraints = {"window_size": [StrOptions({"1D", "2D"})]}

    def __init__(self, window_size: str = "1D") -> None:
        self.window_size = window_size

    def get_feature_names_out(self):
        return ["wip"]

    def _fit(self, X: pd.DataFrame, y=None):
        return self

    def _transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self)
        timestamps = self._timestamps(X)
        case_ids = self._case_ids(X)

        wip_by_window = case_ids.groupby(
            pd.Grouper(freq=self.window_size, level="timestamp")
        ).nunique()
        bins = pd.cut(
            timestamps,
            bins=wip_by_window.index,
            labels=wip_by_window.index[:-1],
        )
        wip = bins.map(wip_by_window).fillna(bins.isna().sum()).values
        return wip
