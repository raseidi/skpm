"""
Event Feature Extraction
========================

In this tutorial, we introduce a few feature extraction techniques
available in our library. Currently, we provide two modules for
feature extraction: :mod:`skpm.case_feature_extraction` and
:mod:`skpm.event_feature_extraction`. The former is still
uder development so we will focus on the latter.
"""

# %%
# Event features
# --------------
# The :mod:`skpm.event_feature_extraction` module provides
# a set of function to extract relevant features proposed in the
# literature. In this example, we show how to extract features from
# timestamps, resources, and an the inter-case perspective.
#
# Time-related features
# ---------------------
# The :class:`skpm.event_feature_extraction.TimestampExtractor` class
# allows us to extract several features, such as the execution time of
# each event, the accumulated time throughout the case, and the weekday.
# Let's see how it works.

# %%
from skpm.config import EventLogConfig as elc
from skpm.event_feature_extraction import TimestampExtractor
from skpm.event_logs import split, BPI17

# download the dataset
log = BPI17()

# select the columns of interest
df = log.dataframe[[elc.case_id, elc.activity, elc.timestamp, elc.resource]]

# split the data into train and test
train, test = split.unbiased(df, **log.unbiased_split_params)

# extract the features
te = TimestampExtractor().fit(train)
train[te.get_feature_names_out()] = te.transform(train)
test[te.get_feature_names_out()] = te.transform(test)

# first event as an example
train.iloc[0, :].T

# %%
# In the literature, features like the weekday are usually extracted
# as a categorical features, but we currently implement it as a
# numerical by normalizing the values between `[-0.5, 0.5]`.
# In the future, we intend to provide a parameter to choose between
# the two options.
# 
# Resource-related features
# -------------------------
# The resource pool extractor is a feature extractor that identifies
# resource roles based on the correlation between activities and resources.
# You can easily use this function as shown below:

from skpm.event_feature_extraction import ResourcePoolExtractor

re = ResourcePoolExtractor().fit(train)
# re.get_feature_names_out()
train["resource_role"] = re.transform(train)
test["resource_role"] = re.transform(test)

train.loc[0, [elc.case_id, elc.activity, elc.resource, "resource_role"]].T

# %%
# From the machine learning perspective, it can be seen as a nice way
# to encode the resource information and reduce the dimensionality of the
# data. In this example, we grouped 133 resource labels into 5 roles:

import matplotlib.pyplot as plt
plt.style.use("ggplot")

features = train[[elc.resource, "resource_role"]].nunique().index.values
values = train[[elc.resource, "resource_role"]].nunique().values

fig, ax = plt.subplots()
ax.bar(features, values, edgecolor="black")

# %%
# Inter-case features
# -------------------
# Inter-case features refer to features that are computed based on the
# relationship between different cases. It aims to quantify and module
# the resource sharing between cases, for instance. In the current version
# of our library, we only have a simple example of such feature: the number of
# cases in progress simultaneously. This feature is commonly called
# work in progress.
#
# Let's see how it works:

from skpm.event_feature_extraction import WorkInProgress

wip = WorkInProgress()
wip.fit(train)
train["wip"] = wip.transform(train)
test["wip"] = wip.transform(test)

# visualizing it
(
    train
    .set_index(elc.timestamp)
    .resample("D")["wip"].mean()
    .plot(
        y="wip", 
        kind="line", 
        figsize=(10, 5), 
        title="Average daily \nWork in Progress (WIP) over time")
)

# %%
# In this tutorial, we showed how to extract features from timestamps,
# resources, and the inter-case perspective. We hope you find it useful
# for your projects. If you have any questions or suggestions, please
# open an issue on our GitHub repository or
# `contact me <https://raseidi.github.io/pages/contact.html>`_ directly.