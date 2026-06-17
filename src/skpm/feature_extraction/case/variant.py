from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from skpm.base import CaseLevelTransformer


class VariantExtractor(CaseLevelTransformer):
    """Extract trace variants from an event log.

    A trace variant is the ordered tuple of activities of a case. This is a
    **case-level** transformer: it emits one row per case (see
    :class:`~skpm.base.CaseLevelTransformer`), so it is a terminal step rather
    than a Pipeline intermediate.
    """

    _parameter_constraints: dict = {}

    def __init__(self, strategy: str = "default"):
        self.strategy = strategy

    def _fit(self, X, y=None):
        if self.strategy != "default":
            raise NotImplementedError("Only the default strategy is supported.")

        variants = (
            X.groupby(level="case_id", sort=False, observed=True)[self.activity]
            .apply(tuple)
            .rename("variant")
            .to_frame()
            .reset_index()
        )
        self._le = LabelEncoder()
        variants["variant"] = self._le.fit_transform(variants["variant"])
        self.variants_ = variants
        return self

    def _transform(self, X, y=None):
        check_is_fitted(self, "variants_")
        # Case-level artifact computed at fit; one row per case.
        return self.variants_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "variants_")
        return list(self.variants_.columns)

    def inverse_transform(self, X):
        check_is_fitted(self, "_le")
        return self._le.inverse_transform(X)
