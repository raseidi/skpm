"""Simple, leakage-free baselines for event-level process targets."""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from skpm.base import BaseProcessEstimator

__all__ = ["ActivityMeanRegressor"]


class ActivityMeanRegressor(RegressorMixin, BaseProcessEstimator):
    """Predict each event's target as the training mean of its activity label.

    A strong, leakage-free baseline for event-level targets such as remaining
    time: the per-activity mean captures *where in the process* an event is,
    which on many logs carries most of the signal. Activities unseen at fit
    time fall back to the global training mean.

    Input is an event log (flat or canonical); the ``activity`` column is read
    via the configured canonical name. It is a scikit-learn regressor, so it
    slots into a :class:`~sklearn.pipeline.Pipeline` as the final estimator
    (with ``passthrough`` feature/encoding steps) for a head-to-head comparison
    with learned models under the same cross-validation.

    Attributes
    ----------
    activity_means_ : pandas.Series
        Mean target per activity seen at fit time.
    global_mean_ : float
        Mean target over all fit events (fallback for unseen activities).
    """

    _parameter_constraints: dict = {}

    def fit(self, X, y):
        X = self._validate_log(X)
        self._record_feature_names_in(X)
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.activity_means_ = y.groupby(X[self.activity], observed=True).mean()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X):
        check_is_fitted(self, "activity_means_")
        X = self._validate_log(X, copy=False)
        preds = X[self.activity].map(self.activity_means_)
        return preds.fillna(self.global_mean_).to_numpy()
