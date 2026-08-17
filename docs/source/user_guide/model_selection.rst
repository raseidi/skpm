.. _model_selection:

=========================
Splitting and validation
=========================

.. currentmodule:: skpm.model_selection

Splitting comes first in a SkPM workflow — before feature extraction, before
fitting anything.

.. _train_test_split:

``train_test_split``
====================

:func:`train_test_split` is the documented entry point. It splits **by whole
cases**, so no case ever has some of its events in training and others in test:

.. code-block:: python

   from skpm.model_selection import train_test_split

   train, test = train_test_split(log, test_size=0.2)

It returns **two event logs**, not ``(X_train, X_test, y_train, y_test)``.
Which target to predict is a modelling decision — remaining time is a
regression, next activity a classification, and the same split also serves
unsupervised work — so the target is yours to build, once per side
(see :ref:`targets`).

They are named ``train``/``test`` rather than ``X_train``/``X_test`` on
purpose. They are raw event logs with string activities, not feature matrices.

.. note::

   Splitting *after* computing a target gives identical values to splitting
   before. Both strategies are case-level and all three targets are case-local,
   so nothing crosses a case boundary. The leakage that is real lives on the
   ``X`` side: :class:`~skpm.feature_extraction.WorkInProgress` reads across
   cases, and :class:`~skpm.sequence_encoding.Aggregation`,
   :class:`~skpm.feature_extraction.ResourcePoolExtractor` and
   :class:`~skpm.sequence_encoding.Bucketing` are *fitted*. Only fitting the
   pipeline on ``train`` alone prevents it.

.. _temporal_split:

The temporal strategy
=====================

``strategy="temporal"`` is the default. It cuts the log at a timestamp derived
from ``test_size`` and needs no other parameters, so it works on any log:

.. code-block:: python

   train, test = train_test_split(pd.read_csv("my_log.csv"))

Here ``test_size`` is a fraction of the log's **calendar span**, not of its
cases. If arrivals are uneven — a quiet December, a busy March — the resulting
case counts are uneven too. Check them rather than assuming:

.. code-block:: python

   from skpm import case_ids

   case_ids(train).nunique(), case_ids(test).nunique()

.. _unbiased_split:

The unbiased strategy
=====================

An event log covers a limited recording window, and that window distorts the
data at both ends. Cases already running when recording started are missing
their beginning; cases still running when it stopped are missing their end, so
their durations — and therefore any remaining-time label — are wrong.

``strategy="unbiased"`` drops those cases and separates train and test in time,
following Weytjens and De Weerdt [1]_:

.. code-block:: python

   from skpm.event_logs import BPI17

   train, test = train_test_split(BPI17(), strategy="unbiased")

It is parameterised by ``start_date``, ``end_date`` (bounds on case start and
end, at monthly resolution) and ``max_days`` (the longest case duration kept).
These are constants published with the benchmark rather than quantities derived
from the data, so SkPM never guesses them. The six loaders that ship them
(:class:`~skpm.event_logs.BPI12`, :class:`~skpm.event_logs.BPI17`,
:class:`~skpm.event_logs.BPI19`,
:class:`~skpm.event_logs.BPI20PrepaidTravelCosts`,
:class:`~skpm.event_logs.BPI20RequestForPayment`,
:class:`~skpm.event_logs.BPI20TravelPermitData`) supply them automatically —
which is why the call above carries no dates.

Precedence is explicit keyword → loader → error naming what is missing. For a
log of your own, pass them yourself; ``start_date`` may stay ``None``, as it is
for most published logs:

.. code-block:: python

   train, test = train_test_split(
       my_log, strategy="unbiased", end_date="2017-01", max_days=47.81,
   )

For ``"unbiased"``, ``test_size`` is a proportion of **cases**. Passing an
unbiased-only parameter together with ``strategy="temporal"`` raises rather
than being silently ignored.

Both strategies are also available directly as
:func:`~skpm.event_logs.split.temporal` and
:func:`~skpm.event_logs.split.unbiased`, which take their own parameters and
nothing else.

.. _cross_validation:

Cross-validation
================

Use scikit-learn's, with one requirement: **folds must not split a case**.
:class:`~sklearn.model_selection.GroupKFold` does that when you pass the case
identifiers as groups:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV, GroupKFold
   from skpm import case_ids

   search = GridSearchCV(
       pipeline,
       param_grid={"model": [LinearRegression(), RandomForestRegressor()]},
       scoring="neg_mean_absolute_error",
       cv=GroupKFold(n_splits=4),
   )
   search.fit(train, y_train, groups=case_ids(train))

Without ``groups``, a plain :class:`~sklearn.model_selection.KFold` would put a
case's early events in a training fold and its later events in the validation
fold — and the model would validate against cases it has already partly seen.

Pass a DataFrame, not an ``EventLog``: meta-estimators index ``X`` before any
SkPM code runs. ``train_test_split`` already returns DataFrames.

.. topic:: References

   .. [1] Hans Weytjens and Jochen De Weerdt, "Creating Unbiased Public
      Benchmark Datasets with Data Leakage Prevention for Predictive Process
      Monitoring", *Business Process Management Workshops*, 2021.

.. minigallery:: skpm.train_test_split skpm.model_selection.train_test_split
   :add-heading: Examples
