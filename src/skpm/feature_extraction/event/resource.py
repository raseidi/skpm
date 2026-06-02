import warnings

import numpy as np
from pandas import DataFrame
from scipy.sparse.csgraph import connected_components
from sklearn.base import check_is_fitted
from sklearn.discriminant_analysis import Interval, Real

from skpm.base import BaseProcessTransformer
from skpm.warnings import ConceptDriftWarning


class ResourcePoolExtractor(BaseProcessTransformer):
    """Extract resource roles from resource-activity correlations.

    The transformer reads the configured ``activity`` and ``resource``
    columns as features and assigns each event the role of its resource.
    It does not look at ``case_id`` or ``timestamp``, so it works with
    any event-log MultiIndex but does not modify it either.

    Parameters
    ----------
    threshold : float, default=0.7
        Correlation threshold for grouping resources into the same role.

    References
    ----------
    [1] Minseok Song, Wil M.P. van der Aalst. "Towards comprehensive
        support for organizational mining," Decision Support Systems (2008).
    [2] Adapted from https://github.com/AdaptiveBProcess/GenerativeLSTM
    """

    _parameter_constraints = {
        "threshold": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def get_feature_names_out(self):
        return ["resource_roles"]

    def _fit(self, X: DataFrame, y=None):
        X = self._validate_log(X, copy=True)
        self._check_feature_columns(X)

        self.atoi_, self.itoa_ = self._define_vocabs(X[self.activity].unique())
        self.rtoi_, self.itor_ = self._define_vocabs(X[self.resource].unique())

        activity = X[self.activity].map(self.atoi_)
        resource = X[self.resource].map(self.rtoi_)

        freq_matrix = (
            DataFrame({"activity": activity, "resource": resource})
            .groupby(["activity", "resource"])
            .value_counts()
            .to_dict()
        )

        profiles = np.zeros((len(self.rtoi_), len(self.atoi_)), dtype=int)
        for (a, r), freq in freq_matrix.items():
            profiles[r, a] = freq

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(profiles)

        np.fill_diagonal(corr, 0)

        n_components, labels = connected_components(
            corr > self.threshold, directed=False
        )

        self.resource_to_roles_ = {
            user_id: role_ix
            for role_ix in range(n_components)
            for user_id in np.where(labels == role_ix)[0]
        }
        return self

    def _transform(self, X: DataFrame, y=None):
        check_is_fitted(self, "resource_to_roles_")
        X = self._validate_log(X, copy=False)
        self._check_feature_columns(X)

        resource = self._check_unknown(
            X[self.resource].values, self.rtoi_.keys(), self.resource
        )
        return DataFrame({self.resource: resource}, index=X.index)[self.resource].map(
            self.rtoi_
        ).map(self.resource_to_roles_).values

    def _check_feature_columns(self, X: DataFrame) -> None:
        missing = [
            col for col in (self.activity, self.resource) if col not in X.columns
        ]
        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}. "
                f"Configure them via EventLogConfig.set_global_config."
            )
        if X[self.activity].isnull().any():
            raise ValueError("Activity column contains null values.")
        if X[self.resource].isnull().any():
            raise ValueError("Resource column contains null values.")

    def _check_unknown(self, input: np.ndarray, vocab, name: str) -> np.ndarray:
        unknown = set(input) - set(vocab)
        if unknown:
            warnings.warn(
                f"The label '{name}' contains values unseen during fitting. "
                f"These values will be set to 'UNK': {unknown}",
                category=ConceptDriftWarning,
                stacklevel=2,
            )
        return np.array(["UNK" if x in unknown else x for x in input])

    def _define_vocabs(self, unique_labels: np.ndarray):
        stoi, itos = {"UNK": 0}, {0: "UNK"}
        stoi.update({label: i + 1 for i, label in enumerate(unique_labels)})
        itos.update({i + 1: label for i, label in enumerate(unique_labels)})
        return stoi, itos
