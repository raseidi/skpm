import inspect
from typing import Optional, Union

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)

from skpm.config import EventLogConfig as elc
from skpm.feature_extraction.case.time import TimestampCaseLevel
from skpm.feature_extraction.event.time import TimestampEventLevel
from skpm.utils import validate_columns, validate_methods_from_class

def _to_list(x):
    if x == "all" or x is None:
        return x
    return [x] if not isinstance(x, list) else x

class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Extracts features from a timestamp column.

    This class extracts various features and targets from a timestamp column in a DataFrame.
    
    The current targets are: `execution_time` and `remaining_time`. All the remaining attributes
    are considered as features.

    Parameters:
    -----------
        case_features (Union[list, str], optional): List of case features to extract. Defaults to "all".
        event_features (Union[list, str], optional): List of event features to extract. Defaults to "all".
        targets (Union[list, str], optional): List of target features to extract. Defaults to None.
        time_unit (str, optional): Time unit for the features. Defaults to "s".
        

    Attributes:
    -----------
        _n_features_out: int
            Number of features extracted.
        _n_targets_out: int
            Number of targets extracted.

    Methods:
    --------
        fit(X, y=None):
            Fit the transformer to the input data.
        transform(X, y=None):
            Transform the input data to calculate timestamp features.
        get_feature_names_out():
            Get the names of the features extracted.
        inverse_transform(X):
            Inverse transform the input data.

    Notes:
    ------
    - This class requires a DataFrame with columns for case IDs and timestamps.
    - Validation of columns and timestamps is performed in the `fit` method.
    - Lowest scale is seconds. Nanoseconds, milliseconds, etc. are disregarded.

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
    available_targets = ["execution_time", "remaining_time"]

    def __init__(
        self,
        case_features: Union[str, list, None] = "all",
        event_features: Union[str, list, None] = "all",
        targets: Optional[Union[str, list]] = None,
        time_unit: str = "s",
    ):
        # TODO: feature time unit (secs, hours, days, etc)
        # TODO: subset of features rather than all
        # TODO: param for event-level and case-level

        self.case_features = _to_list(case_features)
        self.event_features = _to_list(event_features)
        self.targets = _to_list(targets)
        self.time_unit = time_unit

    def fit(
        self,
        X: pd.DataFrame,
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

        self.event_features = validate_methods_from_class(
            class_obj=TimestampEventLevel, 
            methods=self.event_features
        )
        self.case_features = validate_methods_from_class(
            class_obj=TimestampCaseLevel,
            methods=self.case_features, 
        )
        self.targets = validate_methods_from_class(
            class_obj=TimestampCaseLevel,
            methods=self.targets, 
        )
        
        self._n_features_out = len(self.event_features) + len(
            self.case_features
        )
        self._n_targets_out = len(self.targets)
        
        if self._n_features_out + self._n_targets_out == 0:
            raise ValueError("No features selected. Please select at least one feature, either from the event level or the case level.")

        return self

    def get_feature_names_out(self):
        return [
            f[0] for f in self.case_features + self.event_features + self.targets
        ]

    def transform(self, X: pd.DataFrame, y=None):
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

        # case-level features
        self.group_ = X.groupby(
            elc.case_id, as_index=False, group_keys=False, observed=True
        )
        for feature_name, feature_fn in self.case_features:
            X[feature_name] = feature_fn(
                case=self.group_,
                ix_list=X.index.values,
                time_unit=self.time_unit,
            )

        # for event-level features
        for feature_name, feature_fn in self.event_features:
            sig = inspect.signature(feature_fn)
            if "time_unit" in sig.parameters:
                X[feature_name] = feature_fn(X[elc.timestamp], time_unit=self.time_unit)
            else:
                X[feature_name] = feature_fn(X[elc.timestamp])

        # targets
        for feature_name, feature_fn in self.targets:
            X[feature_name] = feature_fn(
                case=self.group_,
                ix_list=X.index.values,
                time_unit=self.time_unit,
            )
        output_columns = [
            feature[0]
            for feature in self.case_features + self.event_features + self.targets
        ]
        return X.loc[:, output_columns].values

    def _validate_data(self, X: pd.DataFrame):
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
        assert isinstance(X, pd.DataFrame), "Input must be a dataframe."
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
        self, x: pd.DataFrame, timestamp_format: str = "%Y-%m-%d %H:%M:%S"
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

