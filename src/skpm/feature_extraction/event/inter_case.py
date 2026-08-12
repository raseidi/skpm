import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class WorkInProgress(BaseProcessTransformer):
    """Work in Progress (WIP) feature extractor.

    Partitions time into consecutive fixed windows of ``window_size`` and
    labels each event with the number of distinct cases active (i.e. having
    at least one event) in the event's window. Input is an event-log
    DataFrame with the canonical event-log MultiIndex
    (``case_id``, ``timestamp``, ``event_id``).

    Parameters
    ----------
    window_size : str, default='1D'
        Pandas offset alias describing the window width
        (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        Invalid aliases raise a ``ValueError`` at fit time.

    Returns
    -------
    ndarray of shape (n_samples,)
    """

    _parameter_constraints = {"window_size": [str]}

    def __init__(self, window_size: str = "1D") -> None:
        self.window_size = window_size

    def get_feature_names_out(self):
        return ["wip"]

    def _fit(self, X: pd.DataFrame, y=None):
        try:
            to_offset(self.window_size)
        except ValueError as err:
            raise ValueError(
                f"window_size={self.window_size!r} is not a valid pandas "
                "offset alias."
            ) from err
        return self

    def _transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self)
        if X.empty:
            raise ValueError("Cannot compute WIP on an empty event log.")
        case_ids = self._case_ids(X)
        wip = case_ids.groupby(
            pd.Grouper(freq=self.window_size, level=self.timestamp)
        ).transform("nunique")
        return wip.values
