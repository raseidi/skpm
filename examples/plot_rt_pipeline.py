"""
Remaining Time Prediction Pipeline
==================================

This example demonstrates how to build a pipeline for remaining time prediction
using the BPI 2013 Closed Problems dataset.

The pipeline consists of the following steps:
1. Preprocessing: Extracts features from the event log.
2. Encoding: Aggregates the extracted features.
3. Regression: Fits a regression model to predict the remaining time.

The pipeline is evaluated using the R^2 score. We conclude by showing a trick
to improve the performance of the regression model by transforming the target
using `sklearn.compose.TransformedTargetRegressor`.
"""

# %%
# Required imports
# ------------------
# We start by importing the required modules and classes.
from sklearn.compose import TransformedTargetRegressor
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from skpm.encoding import Aggregation
from skpm.config import EventLogConfig as elc
from skpm.event_feature_extraction import (
    TimestampExtractor,
    ResourcePoolExtractor,
)
from skpm.event_feature_extraction.targets import remaining_time
from skpm.event_logs import BPI13ClosedProblems

# %%
# Download the example dataset
# ----------------------------
# We can automatically download event logs using SkPM.
# In this example, let's use the :class:`~skpm.event_logs.BPI13ClosedProblems`.
log = BPI13ClosedProblems()
log  # Note: this is a TUEventLog object, not a dataframe

# %%
# Subsequently, let's access the `pd.DataFrame` and
# extract the target variable `remaining_time` using the
# :func:`~skpm.event_feature_extraction.targets.remaining_time` function.
log = log.dataframe.copy()
log = log[[elc.case_id, elc.activity, elc.resource, elc.timestamp]]

# extract the target variable
log.loc[:, "remaining_time"] = remaining_time(log, time_unit="seconds")

# In order to keep this example simple, we are skipping the train-test split.
X_train = log.drop(columns=["remaining_time"])
y_train = log["remaining_time"]

log.head()

# %%
# Build the pipeline
# ------------------
# We build the pipeline by creating a sequence of steps.
# The pipeline consists of the following steps:
#
# 1. **Preprocessing**: Extracts features from the event log.
#
# 2. **Encoding and normalizing**: Aggregates the extracted features and
# applies the StandardScaler.
#
# 3. **Regression**: Fits a regression model to predict the remaining time.
#
# We create a `ColumnTransformer` to apply different transformations to
# different columns. More specifically, we apply the following transformations:
#
# - `TimestampExtractor` to extract timestamp features.
#
# - `OneHotEncoder` to encode the activity column.
#
# - `ResourcePoolExtractor` to extract resource pool of each activity.
transformers = ColumnTransformer(
    transformers=[
        (
            "timestamp_features",
            TimestampExtractor(),
            [elc.timestamp, elc.case_id],
        ),
        (elc.activity, OneHotEncoder(sparse_output=False), [elc.activity]),
        (
            elc.resource,
            ResourcePoolExtractor(),
            [elc.case_id, elc.activity, elc.resource],
        ),
        (elc.case_id, "passthrough", [elc.case_id]),
    ]
)

# %%
# Integrating the preprocessing transformers with the full pipeline.
# The pipeline will transformer/extractu features, encode the traces,
# normalize the features, and fit a regression model to predict the remaining
# time.
pipe = Pipeline(
    [
        ("preprocessing", transformers),
        ("encoding", Aggregation(method="mean")),
        ("scaling", StandardScaler()),
        ("regressor", RandomForestRegressor()),
    ]
)

print(pipe.fit(X_train, y_train).score(X_train, y_train))

# %%
# We can leverage the `TransformedTargetRegressor` class to improve the
# performance of the regression model. This class allows us to transform the
# target variable using a transformer before fitting the model. In this example,
# we use the `FunctionTransformer` class to apply the `log1p` transformation to
# the target. The pipeline will output the target in the original scale since
# we set the `inverse_func` parameter to `np.expm1`.
#
# Such trick allows us to enhance the predictive performance of the model.

# sphinx_gallery_start_ignore
import warnings
warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
y_trans = FunctionTransformer(np.log1p, inverse_func=np.expm1)
regr = TransformedTargetRegressor(regressor=pipe, transformer=y_trans)

print(regr.fit(X_train, y_train).score(X_train, y_train))

# %%
# In this tutorial, we showed how to run an end-to-end predictive
# process monitoring pipleine. We hope you find it useful
# for your projects. If you have any questions or suggestions, please
# open an issue on our GitHub repository or
# `contact me <https://raseidi.github.io/pages/contact.html>`_ directly.