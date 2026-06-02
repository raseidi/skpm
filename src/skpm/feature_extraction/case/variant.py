from sklearn.preprocessing import LabelEncoder

from skpm.base import BaseProcessTransformer
from skpm.feature_extraction.case._helpers import ensure_not_pipeline


class VariantExtractor(BaseProcessTransformer):
    """Extract trace variants from an event log."""

    _parameter_constraints: dict = {}

    def __init__(self, strategy: str = "default"):
        self.strategy = strategy

    @ensure_not_pipeline
    def fit(self, X, y=None):
        if self.strategy != "default":
            raise NotImplementedError("Only the default strategy is supported.")

        X = self._validate_log(X, copy=False)
        self.variants = (
            X.groupby(level="case_id", sort=False, observed=True)[self.activity]
            .apply(tuple)
            .rename("variant")
            .to_frame()
            .reset_index()
        )

        self._le = LabelEncoder()
        self.variants["variant"] = self._le.fit_transform(self.variants["variant"])
        return self

    def transform(self, X):
        """Get trace variants."""
        return self.variants

    def inverse_transform(self, X):
        return self._le.inverse_transform(X)
