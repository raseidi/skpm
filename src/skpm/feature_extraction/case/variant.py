from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from skpm.feature_extraction.case._helpers import ensure_not_pipeline


class VariantExtractor(TransformerMixin, BaseEstimator):
    """Extract trace variants from an event log."""

    def __init__(self, strategy="default"):
        self.strategy = strategy

    @ensure_not_pipeline
    def fit(self, X, y=None):
        if self.strategy != "default":
            raise NotImplementedError("Only the default strategy is supported.")

        self.variants = (
            X.groupby("case:concept:name", as_index=False)["concept:name"]
            .apply(tuple)
            .rename(columns={"concept:name": "variant"})
        )

        self._le = LabelEncoder()
        self.variants["variant"] = self._le.fit_transform(
            self.variants["variant"]
        )
        return self

    def transform(self, X):
        """Get trace variants."""
        return self.variants

    def inverse_transform(self, X):
        """Get trace variants."""
        return self._le.inverse_transform(X)
