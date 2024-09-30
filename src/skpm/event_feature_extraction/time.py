from typing import Union

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)
# from sklearn.utils import check_pandas_support

from skpm.config import EventLogConfig as elc
from skpm.utils import validate_columns, validate_methods_from_class

DataFrame: pd.DataFrame = pd.DataFrame 

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

    def __init__(self, 
                 case_level: Union[str, list] = "all", 
                 event_level: Union[str, list] = "all", 
                 time_unit: str = "secs"):
        # TODO: feature time unit (secs, hours, days, etc)
        # TODO: subset of features rather than all
        # TODO: param for event-level and case-level
        self.features = "all"
        self.case_level = case_level
        self.event_level = event_level
        self.time_unit = time_unit

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
        
        self.event_level_features = validate_methods_from_class(self.features, TimestampEventLevel)
        self.case_level_features = validate_methods_from_class(self.features, TimestampCaseLevel)
        
        self._n_features_out = len(self.event_level_features) + len(self.case_level_features)
        return self

    def get_feature_names_out(self):
        return [f[0] for f in self.case_level_features + self.event_level_features]

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

        # for case-level features
        self.group_ = X.groupby(
            elc.case_id, as_index=False, group_keys=False, observed=True
        )

        kwargs = {
            "case": self.group_,
            "ix_list": X.index.values,
        }

        for feature_name, feature_fn in self.case_level_features:
            X[feature_name] = feature_fn(**kwargs)

        # for event-level features
        for feature_name, feature_fn in self.event_level_features:
            X[feature_name] = feature_fn(X[elc.timestamp])

        output_columns = [feature[0] for feature in self.case_level_features + self.event_level_features]
        return X.loc[:, output_columns].values

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
            # pd = check_pandas_support(
            #     "'pandas' not found. Please install it to use this method."
            # )
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


class TimestampEventLevel:
    """
    Provides methods to extract time-related features from the event level.
    
    Implementing event-level and case-level seperately makes code faster since here we do not need to group by case_id.
    
    """

    @classmethod
    def secs_within_day(cls, X):
        """Extract the number of seconds elapsed within each day from the timestamps encoded as value between [-0.5, 0.5]."""
        return ((X.dt.hour * 3600 + X.dt.minute * 60 + X.dt.second) / 86400) - 0.5
        
    @classmethod
    def week_of_year(cls, X):
        """Week of year encoded as value between [-0.5, 0.5]"""
        return (X.dt.isocalendar().week - 1) / 52.0 - 0.5
    

    @classmethod
    def sec_of_min(cls, X):
        """Minute of hour encoded as value between [-0.5, 0.5]"""    
        return X.dt.second / 59.0 - 0.5

    @classmethod
    def min_of_hour(cls, X):
        """Minute of hour encoded as value between [-0.5, 0.5]"""
    
        return X.dt.minute / 59.0 - 0.5

    @classmethod
    def hour_of_day(cls, X):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        
        return X.dt.hour / 23.0 - 0.5

    @classmethod
    def day_of_week(cls, X):
        """Hour of day encoded as value between [-0.5, 0.5]"""

        
        return X.dt.dayofweek / 6.0 - 0.5

    @classmethod
    def day_of_month(cls, X):
        """Day of month encoded as value between [-0.5, 0.5]"""
        return (X.dt.day - 1) / 30.0 - 0.5

    @classmethod
    def day_of_year(cls, X):
        """Day of year encoded as value between [-0.5, 0.5]"""

        
        return (X.dt.dayofyear - 1) / 365.0 - 0.5

    @classmethod
    def month_of_year(cls, X):
        """Month of year encoded as value between [-0.5, 0.5]"""
        return (X.dt.month - 1) / 11.0 - 0.5

class TimestampCaseLevel:
    """Provides methods to extract time-related features from the case level
    
    Implementing event-level and case-level seperately makes code faster since, since here is slower due to the groupby dependency.
    """
    
    @classmethod
    def execution_time(cls, case, ix_list):
        """Calculate the execution time of each event in seconds."""
        return case[elc.timestamp].diff().loc[ix_list].dt.total_seconds().fillna(0)

    @classmethod
    def accumulated_time(cls, case, ix_list):
        """Calculate the accumulated time from the start of each case in seconds."""
        return (
            case[elc.timestamp]
            .apply(lambda x: x - x.min())
            .loc[ix_list]
            .dt.total_seconds()
        )