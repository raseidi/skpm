from typing import Union

from pandas import DataFrame
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.utils import check_pandas_support

from skpm.utils import validate_columns, validate_features_from_class


class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Extract features from a timestamp column.

    The class needs a column with case ids and a column with timestamps. The validation of the columns is done in the fit method.

    Args:
        case_col: {str}, default='case_id'
            Name of the column containing the case ids.
        time_col: {str}, default='timestamp'
            Name of the column containing the timestamps.
        features (Union[list, str], optional): list of features. Defaults to "all".
    """

    def __init__(
        self,
        case_col: str = "case_id",
        time_col: str = "timestamp",
        features: Union[list, str] = "all",
    ):
        # TODO: time unit (secs, hours, days, etc)
        self.features = features

        self.case_col = case_col
        self.time_col = time_col

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
        self.features_ = validate_features_from_class(self.features, Timestamp)
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
            self.case_col, as_index=False, group_keys=False, observed=True
        )

        # TODO: preprocess features and kwargs for each class
        # method; otherwise, we gotta pass **kwargs to all methods;
        # this is not a problem for now, but it might be in the
        # future since each class method might have different kwargs
        kwargs = {
            "group": self.group_,
            "ix_list": X.index.values,
            "time_col": self.time_col,
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
        x.columns = validate_columns(
            input_columns=x.columns, required=[self.case_col, self.time_col]
        )

        # check if it is a datetime column
        x[self.time_col] = self._validate_timestamp_format(x)

        return x

    def _validate_timestamp_format(
        self, x: DataFrame, timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        if not x[self.time_col].dtype == "datetime64[ns]":
            pd = check_pandas_support(
                "'pandas' not found. Please install it to use this method."
            )
            try:
                # for now, since we are only employing the BPI event logs,
                # we are assuming that the datetime format is '%Y-%m-%d %H:%M:%S'.
                # TODO: validate alternative datetime formats.
                # '%Y-%m-%d %H:%M:%S' format should be mandatory
                x[self.time_col] = pd.to_datetime(x[self.time_col])
            except:
                raise ValueError(
                    f"Column '{self.time_col}' is not a valid datetime column."
                )

        # TODO: ensure datetime format
        # try:
        #     # Attempt to parse the datetime string with the specified format
        #     datetime_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        #     print(f"'{x}' is a valid datetime with the correct format: {datetime_obj}")
        # except ValueError:
        #     print(f"'{x}' is not in the correct format '%Y-%m-%d %H:%M:%S'")
        #     pass
        return x[self.time_col]


class Timestamp:
    @classmethod
    def execution_time(cls, group, ix_list, time_col="timestamp", **kwargs):
        # diff returns a df, so we need to
        # select the column (i.e., as series) to
        # convert to seconds
        return (
            group[time_col].diff().loc[ix_list, time_col].dt.total_seconds().fillna(0)
        )

    @classmethod
    def accumulated_time(cls, group, ix_list, time_col="timestamp", **kwargs):
        return (
            group[time_col].apply(lambda x: x - x.min()).loc[ix_list].dt.total_seconds()
        )

    @classmethod
    def remaining_time(cls, group, ix_list, time_col="timestamp", **kwargs):
        return (
            group[time_col].apply(lambda x: x.max() - x).loc[ix_list].dt.total_seconds()
        )

    @classmethod
    def within_day(cls, X, time_col="timestamp", **kwargs):
        pd = check_pandas_support(
            "'pandas' not found. Please install it to use this method."
        )
        return (
            pd.to_timedelta(X[time_col].dt.time.astype(str)).dt.total_seconds().values
        )
