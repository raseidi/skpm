"""
API for downloading event logs
==============================

This example demonstrates how we can easily download well-known process mining event logs
from the 4TU.Centre for Research Data using the `skpm.event_logs` module.

The `skpm.event_logs` module provides a set of event logs, such as the Sepsis and BPI 2012.
"""

# %%
# The API overview
# ----------------
# Implementing each event log as a class is a design choice that allows us to
# easily manipulate each of them according to their specific characteristics.
# One of the main challenges in process mining is the completely different
# nature of datasets, since
# each of them is composed of very particular business rules.
#
# For instance, an unbiased split of event logs was proposed in [1]. Roughly
# speaking, each event log is splitted based on specific temporal
# characteristics, which is hard coded within each specific event log. You can
# check this feature in :ref:`Unbiased split
# <sphx_glr_auto_examples_unbiased_split.py>`.
#
# Now, let us see how to easily download event logs below.
#
# Downloading the BPI 2013 event log
# ----------------------------------
# The BPI 2013 event log is a well-known event log that contains data about
# closed problems from the Volvo IT Belgium. We can easily download it as
# follows:

from skpm.event_logs import BPI13ClosedProblems

bpi13 = BPI13ClosedProblems() # this will automatically download it
bpi13

# %%
# Notice, the `__repr__`method returns a brief overview of the event log.
# In order to acess the dataframe, just call the `dataframe` attribute.

bpi13.dataframe.head()

# %%
# References
# ----------
# [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring, 2021. doi: 10.1007/978-3-030-94343-1_2