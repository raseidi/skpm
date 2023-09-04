import inspect
from typing import Union
from pandas import DataFrame
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
    check_array,
    check_is_fitted,
)
from sklearn.utils import check_pandas_support

class Timestamp(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    def __init__(
        self,
        case_col="case_id",
        time_col="timestamp",
        features: Union[list, str] = "all",
    ):
        # in the future we can tune time unit by testing, secs, hours, days, etc
        # self.time_unit = time_unit

        self.time_col = time_col
        self.case_col = case_col
        self.features = features

    def fit(self, X: DataFrame, y=None):
        """Fit transformer.

        Parameters
        ----------
            X : {DataFrame} of shape (n_samples, 2)
            y : None.
                Ignored.

        Returns
        -------
            self : object
                Fitted transformer.
        """
        assert isinstance(X, DataFrame), "Input must be a dataframe."
        assert set([self.case_col, self.time_col]).issubset(
            X.columns
        ), "Input must contain a case id column and a timestamp column."
        
        # tuples (name, fn)
        available_features = inspect.getmembers(
            TimestampExtractor, predicate=inspect.ismethod
        )
        self.features_ = []
        if self.features == "all":
            self.features_ = available_features
        else:
            if not isinstance(self.features, str):
                self.features = [self.features]
            for f in available_features:
                if f[0] in self.features:
                    self.features_.append(f)
            
        # validate self.features_
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
        X = self._check_data(X)

        # feature extraction
        
        # groupby is not done in .fit to avoid users to fit with a train set and transform with a test set
        self.group_ = X.groupby(self.case_col, group_keys=False)

        for feature_name, feature_fn in self.features_:
            X[feature_name] = feature_fn(self.group_, X.index, self.time_col)

        # features_ is a list of tuples (name, fn)
        return X.loc[:, [feature[0] for feature in self.features_]]

    def _check_data(self, X: DataFrame):
        # TODO: https://numpy.org/doc/stable/reference/arrays.datetime.html
        # X must be DataFrame (for now) due to the timestamp support.
        # But apparently numpy supports timestamps, so we can use it in the future
        # plus, if we manage to use numpy, we can use numba to speed up the process and avoid pandas overhead

        assert isinstance(X, DataFrame), "Input must be a dataframe."
        X.columns = self._check_columns(X.columns)

        x = X.copy()
        pd = check_pandas_support(
            "'pandas' not found. Please install it to use 'TimeStampExtractor'."
        )
        x[self.time_col] = pd.to_datetime(x[self.time_col])
        return x[[self.case_col, self.time_col]]

    def _check_columns(self, cols):
        # if the transformer is used in a pipeline, the columns are going to be renamed like this
        # case_col = self.case_col.replace("remainder__", "")
        # time_col = self.time_col.replace("remainder__", "")
        # copy cols
        cols = [col.replace("remainder__", "") for col in cols]
        assert set([self.case_col, self.time_col]).issubset(
            cols
        ), "Input must contain a case id column and a timestamp column."
        return cols
        

class TimestampExtractor:
    @classmethod
    def execution_time(cls, group, ix_list, time_col="timestamp"):
        return group[time_col] \
            .diff()\
            .loc[ix_list]\
            # .dt.total_seconds()\
            # .fillna(0)
            
    @classmethod
    def accumulated_time(cls, group, ix_list, time_col="timestamp"):
        return group[time_col] \
            .apply(lambda x: x - x.min())\
            .loc[ix_list]\
            # .dt.total_seconds()
    
    @classmethod
    def remaining_time(cls, group, ix_list, time_col="timestamp"):
        return group[time_col] \
            .apply(lambda x: x.max() - x)\
            .loc[ix_list]\
            # .dt.total_seconds()
            

# https://github.com/AdaptiveBProcess/GenerativeLSTM/blob/master/support_modules/role_discovery.py#L10
