import numpy as np
from sklearn.base import TransformerMixin
from skpm.config import EventLogConfig as elc
from skpm.base import BaseProcessEstimator


class Bucketing(TransformerMixin, BaseProcessEstimator):
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
        - "clustering": The "clustering" strategy, though not implemented in the provided code, would involve
        using clustering techniques to group events based on similarity or patterns in their attributes. This approach
        could potentially identify clusters of events that exhibit similar behavior or characteristics, allowing for
        more nuanced and context-aware bucketing.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer.

    transform(X, y=None)
        Transform input data by bucketing traces.

    get_feature_names_out()
        Get the names of the output features.
    """

    def __init__(self, method="single"):
        """
        Initialize Bucketing Transformer.

        Parameters
        ----------
        method : str, optional
            The method used for bucketing traces. Possible values are "single", "prefix", or "clustering".
            Default is "single".
        """
        assert method in [
            "single",
            "prefix",
            "clustering",
        ], f"Invalid method: {method}"
        
        self.method = method

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : array-like or DataFrame
            The input data.
        y : array-like or DataFrame, optional
            The target data. Default is None.

        Returns
        -------
        self : Bucketing
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data by bucketing traces.

        Parameters
        ----------
        X : array-like or DataFrame
            The input data.
        y : array-like or DataFrame, optional
            The target data. Default is None.

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
