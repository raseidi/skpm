import numpy as np
from sklearn.calibration import StrOptions
from skpm.config import EventLogConfig as elc
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
    _parameter_constraints = {"method": [StrOptions({"single", "prefix", "clustering"})]}
    
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
        """
        Transform input data by bucketing traces.

        Parameters
        ----------
        X : array-like or DataFrame
            The input data.

        Returns
        -------
        bucket_labels : array
            An array containing the bucket labels assigned to each event.
        """
        if self.method == "single":
            # For the single method, assign all events to a single bucket.
            bucket_labels = np.array(["b1"] * len(X))
        elif self.method == "prefix":
            # For the prefix method, group events by case ID and assign sequential buckets.
            bucket_labels = (
                X.groupby(elc.case_id)
                .cumcount()
                .apply(lambda x: f"b{x + 1}")
                .values
            )
        elif self.method == "clustering":
            # Clustering method is not implemented yet.
            raise NotImplementedError(
                "Clustering method is not implemented yet"
            )

        return bucket_labels

    def get_feature_names_out(self):
        """
        Get the names of the output features.

        Returns
        -------
        feature_names : list
            A list containing the name of the output feature.
        """
        return ["bucket"]
