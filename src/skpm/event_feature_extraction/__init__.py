"""
Notes on inter-case features:
    Inter-case features are features that can be leveraged by cases in parallel. For instance, the availability of a resource at a time window `t_0` can be represented as a binary variable.
    This brings up an observation, not an issue I believe, that `fit` methods in this module will just return self, and all the logic should be within `transform`. This is due to the temporal splits, expected for temporal process data. 
    We cannot `fit` on the train set and `transform` on the test 
    set, i.e. define the bins based on `freq`, since in a temporal 
    split the test set will have unkown bins. TODO: further explore 
    if this is an issue.
    For TimestampExtractor, if we have on the training set a trace [t_0, ..., t_n] whereas on the test set we have the remaining trace, i.e., [t_{n+1}, ..., t_m], the `accumulated_time` feature should take this info into consideration.
"""

from .time import TimestampExtractor
from .resource import ResourcePoolExtractor
from .inter import WorkInProgress

__all__ = ["TimestampExtractor", "ResourcePoolExtractor", "WorkInProgress"]
