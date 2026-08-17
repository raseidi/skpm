.. _user_guide:

==========
User Guide
==========

SkPM turns an event log into a supervised learning problem. It supplies the
parts scikit-learn has no opinion about — the event-log data structure,
process-aware features, prefix encoding, prediction targets, and time-aware
splitting — as ordinary estimators, so the rest of the workflow is scikit-learn
as you already know it.

A SkPM workflow always has the same five steps, in this order:

.. code-block:: python

   from sklearn.ensemble import HistGradientBoostingRegressor
   from sklearn.pipeline import Pipeline

   from skpm.event_logs import BPI17
   from skpm.feature_extraction import TimestampExtractor
   from skpm.feature_extraction.targets import remaining_time
   from skpm.model_selection import train_test_split
   from skpm.sequence_encoding import Aggregation

   # 1. Represent the log.  Loaders do this for you; `to_event_log` does it
   #    for a DataFrame of your own.
   log = BPI17()

   # 2. Split, before extracting features or fitting anything.
   train, test = train_test_split(log, strategy="unbiased")

   # 3. Build the target, once per side.
   y_train = remaining_time(train, time_unit="h")

   # 4. Extract features and encode prefixes, inside a Pipeline.
   # 5. Fit an estimator on the training cases only.
   Pipeline([
       ("time", TimestampExtractor(time_unit="h")),
       ("prefix", Aggregation(method="mean")),
       ("model", HistGradientBoostingRegressor()),
   ]).fit(train, y_train)

The order is not stylistic. Steps 4 and 5 are *fitted*: an encoder learns its
categories, a scaler learns its means, a model learns everything. Fitting any
of them before the split lets the test cases influence the model, and the score
you report afterwards is then optimistic for a reason that has nothing to do
with the process.

The rest of this guide takes the five steps one at a time.

.. toctree::
   :numbered:
   :maxdepth: 2

   event_logs
   model_selection
   targets
   feature_extraction
   sequence_encoding
   composing
