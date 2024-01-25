from sklearn.base import TransformerMixin

from skpm.config import EventLogConfig as elc

from skpm.base import BaseProcessEstimator


class Bucketing(TransformerMixin, BaseProcessEstimator):
    """Event Bucketing Transformer.

    This class implements a method for bucketing traces

    """

    def __init__(self, method="single"):
        assert method in ["single", "prefix", "clustering"]
        self.method = method

    def fit(X, y=None):
        pass

    def transform(self, X, y=None):
        if self.method == "single":  # TODO: not needed imo
            X["bucket"] = "b1"
        elif self.method == "prefix":
            X["bucket"] = X.groupby(elc.case_id).cumcount().apply(lambda x: f"b{x+1}")
        elif self.method == "clustering":
            raise NotImplementedError

        return X
