from typing import Union
from pandas import DataFrame

from sklearn.utils import check_pandas_support
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
    check_is_fitted,
)

from skpm.utils import check_features


class TimestampExtractor(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    def __init__(
        self,
        case_col: str = "case_id",
        time_col: str = "timestamp",
        features: Union[list, str] = "all",
    ):
        """_summary_

        Args:
            case_col: {str}, default='case_id'
                Name of the column containing the case ids.
            time_col: {str}, default='timestamp'
                Name of the column containing the timestamps.
            features (Union[list, str], optional): list of features. Defaults to "all".
        """
        # TODO: time unit (secs, hours, days, etc)
        self.features = features

        # according to the docs, the constructor should only receive data-independent parameters, i.e. 'tunable parameters'
        # thus, I believe passing the column names are not tunable parameters but the features are
        # still, imo it is cleaner to pass the column names as parameters here. Declare the variable name with a trailing underscore to denote that it is not a tunable parameter
        self.case_col_ = case_col
        self.time_col_ = time_col

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
        self.features_ = check_features(self.features, Timestamp)
        self._n_features_out = len(self.features_)
        return self

    def transform(self, X, y=None):
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
        # Check is fit had been called
        check_is_fitted(self, "_n_features_out")

        # data validation
        X = self._validate_data(X)

        self.group_ = X.groupby(self.case_col_, group_keys=False)

        # TODO: preprocess features and kwargs for each class
        # method; otherwise, we gotta pass **kwargs to all methods;
        # this is not a problem for now, but it might be in the
        # future since each class method might have different kwargs
        kwargs = {
            "group": self.group_,
            "ix_list": X.index,
            "time_col": self.time_col_,
            "X": X,
        }

        # feature extraction
        for feature_name, feature_fn in self.features_:
            X[feature_name] = feature_fn(**kwargs)

        # features_ is a list of tuples (name, fn)
        return X.loc[:, [feature[0] for feature in self.features_]]

    def _validate_data(self, X):
        assert isinstance(X, DataFrame), "Input must be a dataframe."
        assert set([self.case_col_, self.time_col_]).issubset(
            X.columns
        ), "Input must contain a case id column and a timestamp column."
        x = X.copy()
        x.columns = self._check_columns(x.columns)

        # check if it is a datetime column
        if not x[self.time_col_].dtype == "datetime64[ns]":
            pd = check_pandas_support(
                "'pandas' not found. Please install it to use this method."
            )
            try:
                x[self.time_col_] = pd.to_datetime(x[self.time_col_])
            except:
                raise ValueError(
                    f"Column '{self.time_col_}' is not a valid datetime column."
                )
        return x[[self.case_col_, self.time_col_]]

    def _check_columns(self, cols):
        # if the transformer is used in a sklearn pipeline,
        # the columns are going to be renamed like this
        # case_col = self.case_col.replace("remainder__", "")
        # time_col = self.time_col.replace("remainder__", "")
        cols = [col.replace("remainder__", "") for col in cols]
        assert set([self.case_col_, self.time_col_]).issubset(
            cols
        ), "Input must contain a case id column and a timestamp column."
        return cols


class Timestamp:
    @classmethod
    def execution_time(cls, group, ix_list, time_col="timestamp", **kwargs):
        return group[time_col].diff().loc[ix_list].dt.total_seconds().fillna(0)

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


# class Resource:
# https://github.com/AdaptiveBProcess/GenerativeLSTM/blob/master/support_modules/role_discovery.py#L10
