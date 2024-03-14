from typing import Union

from pandas import DataFrame
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.utils import check_pandas_support

from skpm.config import EventLogConfig as elc
from skpm.utils import validate_columns, validate_methods_from_class


class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Extracts features from a timestamp column.

       This class extracts various features from a timestamp column in a DataFrame.

       Parameters:
       -----------
           features (Union[list, str], optional): List of features to extract. Defaults to "all".

       Attributes:
       -----------
           features_ : list of tuples
               List of feature names and corresponding feature functions.

       Methods:
       --------
           fit(X, y=None):
               Fit the transformer to the input data.
           transform(X, y=None):
               Transform the input data to calculate timestamp features.

       Notes:
       ------
       - This class requires a DataFrame with columns for case IDs and timestamps.
       - Validation of columns and timestamps is performed in the `fit` method.

       Examples:
       ---------
       >>> from skpm.event_feature_extraction.time import TimestampExtractor
       >>> import pandas as pd
       >>> # Assuming X is your dataframe containing event data with columns 'case_id' and 'timestamp'
       >>> X = pd.DataFrame({'case_id': [1, 1, 2, 2], 'timestamp': ['2023-01-01 10:30:00', '2023-01-01 11:00:00', '2023-01-01 09:00:00', '2023-01-01 09:30:00']})
       >>> feature_extractor = TimestampExtractor()
       >>> feature_extractor.fit(X)
       >>> feature_extractor.transform(X)
       """

    def __init__(
            self,
            features: Union[list, str] = "all",
    ):
        # TODO: time unit (secs, hours, days, etc)
        self.features = features

    def fit(
            self,
            X: DataFrame,
            y=None,
    ):
        """Fit the transformer to the input data.

        This method checks if the input is a DataFrame, validates the required columns,
        and computes the desired features.

        Parameters:
        -----------
        X : DataFrame
            Input DataFrame containing columns for case IDs and timestamps.
        y : None
            Ignored.

        Returns:
        --------
        self : TimestampExtractor
            Fitted transformer instance.
        """
        _ = self._validate_data(X)
        self.features_ = validate_methods_from_class(self.features, Timestamp)
        # del self.features
        self._n_features_out = len(self.features_)
        return self

    def get_feature_names_out(self):
        return [f[0] for f in self.features_]

    def transform(self, X: DataFrame, y=None):
        """Transform the input data to calculate timestamp features.

        Parameters:
        -----------
        X : DataFrame
            Input DataFrame containing columns for case IDs and timestamps.
        y : None
            Ignored.

        Returns:
        --------
        X_tr : DataFrame
            Transformed DataFrame with calculated timestamp features added.
        """
        # Check if fit had been called
        check_is_fitted(self, "_n_features_out")

        # data validation
        X = self._validate_data(X)

        self.group_ = X.groupby(
            elc.case_id, as_index=False, group_keys=False, observed=True
        )

        # TODO: preprocess features and kwargs for each class
        # method; otherwise, we gotta pass **kwargs to all methods;
        # this is not a problem for now, but it might be in the
        # future since each class method might have different kwargs
        kwargs = {
            "case": self.group_,
            "ix_list": X.index.values,
            "X": X,
        }

        for feature_name, feature_fn in self.features_:
            X[feature_name] = feature_fn(**kwargs)

        # features_ is a list of tuples (name, fn)
        # TODO: TimestampExtractor().fit().get_feature_names_out()
        return X.loc[:, [feature[0] for feature in self.features_]].values

    def _validate_data(self, X: DataFrame):
        """
        Validates the input DataFrame and timestamp column.

        Parameters:
        -----------
        X : DataFrame
           Input DataFrame containing columns for case IDs and timestamps.

        Returns:
        --------
        X : DataFrame
           Validated DataFrame after processing.
        """
        assert isinstance(X, DataFrame), "Input must be a dataframe."
        x = X.copy()
        x.reset_index(drop=True, inplace=True)
        # x.columns = self._validate_columns(x.columns)
        valid_cols = validate_columns(
            input_columns=x.columns, required=[elc.case_id, elc.timestamp]
        )
        x = x[valid_cols]

        # check if it is a datetime column
        x[elc.timestamp] = self._validate_timestamp_format(x)

        return x

    def _validate_timestamp_format(
            self, x: DataFrame, timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        """
        Validates the format of the timestamp column.

        Parameters:
        -----------
        x : DataFrame
            DataFrame containing columns for case IDs and timestamps.
        timestamp_format : str, optional
            Expected format of the timestamp, by default "%Y-%m-%d %H:%M:%S".

        Returns:
        --------
        x[elc.timestamp] : Series
            Series containing the validated timestamps.
        """
        if not x[elc.timestamp].dtype == "datetime64[ns]":
            pd = check_pandas_support(
                "'pandas' not found. Please install it to use this method."
            )
            try:
                # for now, since we are only employing the BPI event logs,
                # we are assuming that the datetime format is '%Y-%m-%d %H:%M:%S'.
                # TODO: validate alternative datetime formats.
                # '%Y-%m-%d %H:%M:%S' format should be mandatory
                x[elc.timestamp] = pd.to_datetime(
                    x[elc.timestamp], format=timestamp_format
                )
            except:
                raise ValueError(
                    f"Column '{elc.timestamp}' is not a valid datetime column."
                )

        # TODO: ensure datetime format
        # try:
        #     # Attempt to parse the datetime string with the specified format
        #     datetime_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        #     print(f"'{x}' is a valid datetime with the correct format: {datetime_obj}")
        # except ValueError:
        #     print(f"'{x}' is not in the correct format '%Y-%m-%d %H:%M:%S'")
        #     pass
        return x[elc.timestamp]


class Timestamp:
    """
    Provides various methods for calculating timestamp-related features.

    This class contains static methods for computing timestamp-related features
    from a DataFrame containing event logs.

    Methods:
    --------
    execution_time(case, ix_list, **kwargs):
        Calculate the execution time of each event in seconds.

    accumulated_time(case, ix_list, **kwargs):
        Calculate the accumulated time from the start of each case in seconds.

    remaining_time(case, ix_list, **kwargs):
        Calculate the remaining time until the end of each case in seconds.

    within_day(X, **kwargs):
        Extract the number of seconds elapsed within each day from the timestamps.

    Notes:
    ------
    - These methods operate on pandas DataFrame columns containing timestamps.
    - Some methods may require the 'pandas' library to be installed for execution.

    Examples:
    ---------
    >>> from skpm.event_feature_extraction.time import Timestamp
    >>> import pandas as pd
    >>> # Assuming X is your dataframe containing event data with columns 'case_id' and 'timestamp'
    >>> X = pd.DataFrame({'case_id': [1, 1, 2, 2], 'timestamp': ['2023-01-01 10:30:00', '2023-01-01 11:00:00', '2023-01-01 09:00:00', '2023-01-01 09:30:00']})
    >>> # Calculate execution time for each event
    >>> execution_times = Timestamp.execution_time(X, X.index.values)
    >>> print(execution_times)
    [  0. 1800.   0. 1800.]
    """

    @classmethod
    def execution_time(cls, case, ix_list, **kwargs):
        """Calculate the execution time of each event in seconds.

        Parameters:
        -----------
        case : pandas.Series
            Series containing the timestamps of events for a single case.
        ix_list : array-like
            List of indices corresponding to the events to compute execution time for.

        Returns:
        --------
        execution_times : numpy.ndarray
            Array containing the execution time of each event in seconds.
        """
        return case[elc.timestamp].diff().loc[ix_list].dt.total_seconds().fillna(0)

    @classmethod
    def accumulated_time(cls, case, ix_list, **kwargs):
        """Calculate the accumulated time from the start of each case in seconds.

        Parameters:
        -----------
        case : pandas.Series
            Series containing the timestamps of events for a single case.
        ix_list : array-like
            List of indices corresponding to the events to compute accumulated time for.

        Returns:
        --------
        accumulated_times : numpy.ndarray
            Array containing the accumulated time from the start of each case in seconds.
        """
        return (
            case[elc.timestamp]
            .apply(lambda x: x - x.min())
            .loc[ix_list]
            .dt.total_seconds()
        )

    @classmethod
    def remaining_time(cls, case, ix_list, **kwargs):
        """Calculate the remaining time until the end of each case in seconds.

        Parameters:
        -----------
        case : pandas.Series
            Series containing the timestamps of events for a single case.
        ix_list : array-like
            List of indices corresponding to the events to compute remaining time for.

        Returns:
        --------
        remaining_times : numpy.ndarray
            Array containing the remaining time until the end of each case in seconds.
        """
        return (
            case[elc.timestamp]
            .apply(lambda x: x.max() - x)
            .loc[ix_list]
            .dt.total_seconds()
        )

    @classmethod
    def within_day(cls, X, **kwargs):
        """Extract the number of seconds elapsed within each day from the timestamps.

        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame containing timestamps of events.

        Returns:
        --------
        seconds_within_day : numpy.ndarray
            Array containing the number of seconds elapsed within each day from the timestamps.
        """
        pd = check_pandas_support(
            "'pandas' not found. Please install it to use this method."
        )
        return (
            pd.to_timedelta(X[elc.timestamp].dt.time.astype(str))
            .dt.total_seconds()
            .values
        )
