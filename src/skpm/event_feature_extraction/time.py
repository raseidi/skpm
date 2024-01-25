from typing import Union

from pandas import DataFrame
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin, check_is_fitted)
from sklearn.utils import check_pandas_support

from skpm.config import EventLogConfig as elc
from skpm.utils import validate_columns, validate_methods_from_class


class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Extract features from a timestamp column.

    The class needs a column with case ids and a column with timestamps. The validation of the columns is done in the fit method.

    Args:
        case_col: {str}, default='case_id'
            Name of the column containing the case ids.
        elc.timestamp: {str}, default='timestamp'
            Name of the column containing the timestamps.
        features (Union[list, str], optional): list of features. Defaults to "all".
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
        """Fit transformer.
        Checks if the input is a dataframe, if it
        contains the required columns, validates
        the timestamp column, and the desired features.

        Parameters
        ----------
            X : {DataFrame} of shape (n_samples, 2)
                The data must contain a column with case ids and
                a column with timestamps.
            y : None.
                Ignored.

        Returns
        -------
            self : object
                Fitted transformer.
        """
        _ = self._validate_data(X)
        self.features_ = validate_methods_from_class(self.features, Timestamp)
        # del self.features
        self._n_features_out = len(self.features_)
        return self

    def get_feature_names_out(self):
        return [f[0] for f in self.features_]

    def transform(self, X: DataFrame, y=None):
        """Extract features from timestamp column.

        Parameters
        ----------
        X : {dataframe} of shape (n_samples, 2)
            The data must contain a column with case ids and a column with timestamps.

        Returns
        -------
        X_tr : {dataframe} of shape (n_samples, n_features)
            Transformed array.
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
        if not x[elc.timestamp].dtype == "datetime64[ns]":
            pd = check_pandas_support(
                "'pandas' not found. Please install it to use this method."
            )
            try:
                # for now, since we are only employing the BPI event logs,
                # we are assuming that the datetime format is '%Y-%m-%d %H:%M:%S'.
                # TODO: validate alternative datetime formats.
                # '%Y-%m-%d %H:%M:%S' format should be mandatory
                x[elc.timestamp] = pd.to_datetime(x[elc.timestamp])
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
    @classmethod
    def execution_time(cls, case, ix_list, **kwargs):
        return case[elc.timestamp].diff().loc[ix_list].dt.total_seconds().fillna(0)

    @classmethod
    def accumulated_time(cls, case, ix_list, **kwargs):
        return (
            case[elc.timestamp]
            .apply(lambda x: x - x.min())
            .loc[ix_list]
            .dt.total_seconds()
        )

    @classmethod
    def remaining_time(cls, case, ix_list, **kwargs):
        return (
            case[elc.timestamp]
            .apply(lambda x: x.max() - x)
            .loc[ix_list]
            .dt.total_seconds()
        )

    @classmethod
    def within_day(cls, X, **kwargs):
        pd = check_pandas_support(
            "'pandas' not found. Please install it to use this method."
        )
        return (
            pd.to_timedelta(X[elc.timestamp].dt.time.astype(str))
            .dt.total_seconds()
            .values
        )
