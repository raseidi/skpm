
"""
Unbiased Split of Event Logs
============================

In this tutorial we provide an overview of how the unbiased split of event
logs [1] works and how to use it in the `skpm` package.
"""

# %%
# The `biased` split problem
# --------------------------
# In machine learning, standardizing how datasets are split is a common, often
# essential, practice to ensure fair and reproducible results. However, in the
# field of Process Mining, machine learning applications have not consistently
# adopted this practice. Weytjens and De Weerdt's work [1] proposes the first
# significant effort to address this gap.
#
# More specifically, their paper tackles three key challenges:
#
# 1. **Inconsistent Dataset Split**: Different datasets and preprocessing
# methods make it hard to compare research outcomes. Certain preprocessing
# choices can even lead to biased results due to the use of domain knowledge
# that may not be accessible to all researchers.
# 2. **Data Leakage**: Training and test sets often overlap, with events from
# the same case appearing in both, which leads to overfitted performance
# measures and inaccurate predictions.
# 3. **Test Set Bias**: The test sets frequently suffer from bias due to
# unequal distributions of case durations and active cases, especially at the
# start and end of the dataset. This skews evaluation results, making them
# less reflective of real-world performance.
#
# The `SkPM` package adapted the available code from the authors' GitHub [2].
#
# Unbised Split API
# -----------------
# Only a few datasets are currently supported by the unbiased split method.
# The usage is really simple and can be seen in the following example:

from skpm.event_logs import split, BPI20RequestForPayment

bpi20 = BPI20RequestForPayment()

train, test = split.unbiased(bpi20, **bpi20.unbiased_split_params)
train.shape, test.shape

# %%
# The hyperparameters for the unbiased split are hardcoded in the original
# implementation: `start_date`, `end_date`, `max_days`. However, they are
# derived based on an data-driven analysis. In the future, we may consider
# to implement this generic approach in order to extend the unbiased split
# to other datasets. The hardcoded hyperparameters are:
#
# - `start_date`: the start date of the event log.
# - `end_date`: the end date of the event log.
# - `max_days`: the maximum duration of cases.
#
bpi20.unbiased_split_params

# %%
# For datasets without hardcoded hyperparameters, an exception will be raised:

from skpm.event_logs import Sepsis

sepsis = Sepsis()
try:
    _ = split.unbiased(sepsis, **sepsis.unbiased_split_params)
except Exception as e:
    print(e)

# %%
# The availalbe datasets are:
#

# - :class:`~src.skpm.event_logs.BPI12`
# - :class:`~skpm.event_logs.BPI12`
# - :class:`~src.skpm.event_logs.bpi.BPI12`
# - :class:`~skpm.event_logs.bpi.BPI12`
# - :class:`BPI17`
# - :class:`BPI19`
# - :class:`BPI20PrepaidTravelCosts`
# - :class:`BPI20TravelPermitData`
# - :class:`BPI20RequestForPayment`
#
# References
# ----------
# [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public Benchmark
# Datasets with Data Leakage Prevention for Predictive Process Monitoring,
# 2021. doi: 10.1007/978-3-030-94343-1_2
# [2] https://github.com/hansweytjens/predictive-process-monitoring-benchmarks