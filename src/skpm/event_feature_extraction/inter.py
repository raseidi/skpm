import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from skpm.config import EventLogConfig as elc


class WorkInProgress(TransformerMixin, BaseEstimator):
    """Work in Progress (WIP) feature extractor.

    This transformer calculates the number of cases (work) in progress within
    specified time windows.

    Parameters:
    -----------
    window_size : str, default='1D'
        Frequency of the time windows to count the number of cases in progress.
        It follows the Pandas offset aliases convention. For more details, see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Returns:
    --------
    ndarray
        WIP feature array of shape (n_samples, 1)

    Examples:
    ---------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from skpm.event_feature_extraction import WorkInProgress
    >>> from skpm.config import EventLogConfig as elc
    >>> # Assuming X is your dataframe containing event data with columns 'timestamp' and 'case_id'
    >>> X = pd.DataFrame({elc.timestamp: pd.date_range(start='2024-01-01', end='2024-01-10', freq='D'),
    ...                     elc.case_id: [1, 1, 2, 3, 4, 4, 4, 5, 6, 6]})
    >>> wip_transformer = WorkInProgress(window_size='2D')  # Calculate WIP over 2-day windows
    >>> wip_transformer.fit_transform(X)
    array([2., 1., 1., 2., 2., 1., 1., 2., 2., 2.])
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
        """Transform the input DataFrame to calculate the Work in Progress (WIP) feature.

        This method calculates the number of cases in progress within specified time windows based on the input event data.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame containing event data with columns 'timestamp' and 'case_id'.

        Returns:
        --------
        ndarray
            WIP feature array of shape (n_samples, 1), where each value represents the number of cases in progress at each time step.

        Notes:
        ------
        1. The method performs the following steps:
            a. Groups the event data by time windows specified by the 'window_size' parameter.
            b. Counts the number of unique cases within each time window.
            c. Maps the counts to the corresponding time windows.
            d. Fills any missing values with the number of NaN values (representing time windows with no events).
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
