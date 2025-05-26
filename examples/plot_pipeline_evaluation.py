"""
Pipeline selection
==================

In this tutorial, we will learn how to choose a suitable pipeline for 
a PPM task. We will compare two approaches for preparing data before 
training Gradient Boosting and Random Forest regressors. The first 
approach uses more detailed steps, including timestamp features, 
one-hot encoding, and resource pool extraction. The second approach is 
simpler and relies only on one-hot encoding. We will train each type 
of model with each approach to see how they perform.
"""

# %%
# Let us first import the necessary libraries and set the random seed 
# for reproducibility.

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from skpm.sequence_encoding import Aggregation
from skpm.config import EventLogConfig as elc
from skpm.event_logs import BPI20PrepaidTravelCosts, split
from skpm.feature_extraction.targets import remaining_time
from skpm.feature_extraction import TimestampExtractor, ResourcePoolExtractor

# Set random state for reproducible results
RANDOM_STATE = 44
np.random.seed(RANDOM_STATE)

# %%
# Below we load one of the BPI20 event logs, select relevant columns 
# for this example, extract the remaining time to use as the target, 
# and split the data into train and test sets.1

# Load event log data
log = BPI20PrepaidTravelCosts()

# Select basic columns
df = log.dataframe[[elc.case_id, elc.activity, elc.resource, elc.timestamp]].copy()
df[elc.timestamp] = pd.to_datetime(df[elc.timestamp], utc=True)

# Compute remaining time in seconds
df["remaining_time"] = remaining_time(df, time_unit="seconds")

# Split into train/test sets using provided split method
train, test = split.unbiased(df, **log.unbiased_split_params)

# Separate features and targets for train and test
X_train = train.drop(columns=["remaining_time"])
y_train = train["remaining_time"]
X_test = test.drop(columns=["remaining_time"])
y_test = test["remaining_time"]

# %%
# Defining an advanced and a simple preprocessing pipeline
# --------------------------------------------------------
# We will define two pipelines for preprocessing the data before
# training the models. 

# Advanced preprocessing pipeline
data_prep_advanced = Pipeline([
    ("preprocessing", ColumnTransformer(
        transformers=[
            ("timestamp_features", TimestampExtractor(), [elc.timestamp, elc.case_id]),
            ("activity_encode", OneHotEncoder(sparse_output=False), [elc.activity]),
            ("resource_pool", ResourcePoolExtractor(), [elc.case_id, elc.activity, elc.resource]),
            ("case_id_pass", "passthrough", [elc.case_id]),
        ])),
    ("encode_agg", Aggregation(method="mean", prefix_len=6)),
    ("scaling", StandardScaler()),
])

data_prep_simple = Pipeline([
    ("preprocessing", ColumnTransformer(
        transformers=[
            ("activity_encode", OneHotEncoder(sparse_output=False), [elc.activity]),
            ("case_id_pass", "passthrough", [elc.case_id]),
    ])),
    ("encode_agg", Aggregation(method="mean", prefix_len=6)),
    ("scaling", StandardScaler()),
])

# %%
# Training the models
# -------------------
# We will train two Gradient Boosting and two Random Forest models
# using the advanced and simple preprocessing pipelines. We will then
# evaluate the models using the root mean squared error (RMSE) metric.

# Gradient Boosting pipelines
gb_pipe_advanced = Pipeline([
    ("preprocessing", data_prep_advanced),
    ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE))
])

gb_pipe_simple = Pipeline([
    ("preprocessing", data_prep_simple),
    ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE))
])

# Random Forest pipelines
rf_pipe_advanced = Pipeline([
    ("preprocessing", data_prep_advanced),
    ("regressor", RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))
])

rf_pipe_simple = Pipeline([
    ("preprocessing", data_prep_simple),
    ("regressor", RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))
])

# %%
# Fit all models:

# Fit all models
gb_pipe_advanced.fit(X_train, y_train)
gb_pipe_simple.fit(X_train, y_train)
rf_pipe_advanced.fit(X_train, y_train)
rf_pipe_simple.fit(X_train, y_train)

# Print scores
print("GB-advanced score:", root_mean_squared_error(y_test, gb_pipe_advanced.predict(X_test)))
print("GB-simple score:", root_mean_squared_error(y_test, gb_pipe_simple.predict(X_test)))
print("RF-advanced score:", root_mean_squared_error(y_test, rf_pipe_advanced.predict(X_test)))
print("RF-simple score:", root_mean_squared_error(y_test, rf_pipe_simple.predict(X_test)))

scores = pd.DataFrame({
    "model": ["GB1", "GB2", "RF1", "RF2"],
    "score": [
        root_mean_squared_error(y_test, gb_pipe_advanced.predict(X_test)),
        root_mean_squared_error(y_test, gb_pipe_simple.predict(X_test)),
        root_mean_squared_error(y_test, rf_pipe_advanced.predict(X_test)),
        root_mean_squared_error(y_test, rf_pipe_simple.predict(X_test))
    ]
})


# %%
# Visualizing the results
# -----------------------
# In this step, we will look at the RMSE scores to understand how well each model performed.
# At first glance, GB1 appears weaker than RF2, which might lead us to believe that Random Forest
# is the better choice. However, this comparison is not fair, because each model was trained using
# a different preprocessing pipeline. To make a fair comparison, we need to examine models that 
# use the same data preparation steps.
#
# When we compare models trained with the same preprocessing pipeline, we see that the Gradient
# Boosting model actually scores better than both Random Forest pipelines. This shows how important 
# it is to evaluate models under consistent preprocessing conditions to accurately judge their 
# performance.

import matplotlib.pyplot as plt
plt.style.use("ggplot")

scores.plot(
    kind="barh", 
    x="model", 
    y="score", 
    color="steelblue", 
    legend=False,
    figsize=(8, 4)
)
plt.ylabel("")
plt.xlabel("RMSE")
plt.xscale("log")
plt.tight_layout()