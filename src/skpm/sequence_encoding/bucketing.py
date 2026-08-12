import pandas as pd
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessTransformer


class Bucketing(BaseProcessTransformer):
    """
    Event Bucketing Transformer inherits from :class:`sklearn.base.TransformerMixin` and :class:`skpm.base.BaseProcessEstimator`.

    This class implements a method for bucketing traces based on different strategies.

    Parameters
    ----------
    method : str, optional
        The method used for bucketing traces. Possible values are "single", "prefix", or "clustering".
        Default is "single".

        - "single": Assigns all events to a single bucket.
        - "prefix": Groups events based on the order in which they occur within each case, assigning sequential buckets.
        - "clustering": Not implemented yet, but intended to assign buckets based on clustering of event features.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer.

    transform(X, y=None)
        Transform input data by bucketing traces.

    get_feature_names_out()
        Get the names of the output features.
    """

    _parameter_constraints = {
        "method": [StrOptions({"single", "prefix", "clustering"})]
    }

    def __init__(self, method="single"):
        """
        Initialize Bucketing Transformer.

        Parameters
        ----------
        method : str, optional
            The method used for bucketing traces. Possible values are "single", "prefix", or "clustering".
            Default is "single".
        """
        self.method = method

    def _transform(self, X, y=None):
        """Transform input data by bucketing traces.

        Returns a DataFrame carrying ``X``'s event-log index, like every other
        transformer here. Returning a bare ndarray instead would leave the
        output index up to scikit-learn's ``set_output`` wrapper, which can
        only recover one when the object handed to ``transform`` is itself a
        DataFrame — so an ``EventLog`` caller would silently get a flat index.
        """
        check_is_fitted(self)
        if self.method == "single":
            return pd.DataFrame({"bucket": ["b1"] * len(X)}, index=X.index)
        if self.method == "prefix":
            # Bucket label = 1-based trace position (b1, b2, ...).
            labels = self._trace_positions(X).map(lambda p: f"b{p + 1}")
            return pd.DataFrame({"bucket": labels}, index=X.index)
        if self.method == "clustering":
            raise NotImplementedError(
                "Clustering method is not implemented yet"
            )
        raise ValueError(f"Unknown bucketing method: {self.method}")

    def get_feature_names_out(self):
        """
        Get the names of the output features.

        Returns
        -------
        feature_names : list
            A list containing the name of the output feature.
        """
        return ["bucket"]
