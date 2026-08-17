.. _composing:

===========================
Composing with scikit-learn
===========================

SkPM transformers are scikit-learn transformers. There is no SkPM pipeline, no
SkPM cross-validator and no SkPM metric — you use scikit-learn's, and this page
is only about how the pieces fit.

.. _pipeline_shape:

The standard pipeline
=====================

A predictive process monitoring pipeline has three parts in a fixed order:
extract features, encode the prefix, fit a model.

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from sklearn.ensemble import HistGradientBoostingRegressor
   from sklearn.pipeline import FeatureUnion, Pipeline
   from sklearn.preprocessing import OneHotEncoder

   from skpm.feature_extraction import TimestampExtractor
   from skpm.sequence_encoding import Aggregation

   features = FeatureUnion([
       ("time", TimestampExtractor(case_features="all",
                                   event_features=None, time_unit="h")),
       ("activity", ColumnTransformer(
           [("one_hot", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), ["activity"])],
           remainder="drop", verbose_feature_names_out=False,
       )),
   ]).set_output(transform="pandas")

   pipeline = Pipeline([
       ("features", features),
       ("prefix", Aggregation(method="mean")),
       ("model", HistGradientBoostingRegressor(random_state=0)),
   ])

   pipeline.fit(train, y_train)

Two details do the work here.

The :class:`~sklearn.pipeline.FeatureUnion` runs the timestamp features and the
one-hot activity encoding side by side, because they read different parts of
the log. The :class:`~sklearn.compose.ColumnTransformer` exists to select the
``activity`` column for one-hot encoding; ``verbose_feature_names_out=False``
keeps the resulting names readable.

The order ``features → prefix → model`` matters:
:class:`~skpm.sequence_encoding.Aggregation` needs numbers, and the one-hot
step is what produces them. Aggregating after one-hot encoding is also what
turns the indicators into activity frequencies over the prefix.

.. _index_propagation:

Where the index goes
====================

SkPM calls ``sklearn.set_config(transform_output="pandas")`` on import, so
every transformer returns a DataFrame and the ``(case_id, timestamp,
event_id)`` index propagates from end to end. Call ``.set_output(transform=
"pandas")`` on scikit-learn containers you build yourself, as above.

You can inspect exactly what the final estimator receives by slicing the
pipeline:

.. code-block:: python

   pipeline[:-1].transform(train).head()

That numeric table, plus ``y``, is the whole contract. Anything downstream of
it is ordinary tabular machine learning.

.. _baselines:

Baselines
=========

An error in hours means nothing without a reference.
:class:`~skpm.baselines.ActivityMeanRegressor` predicts, for each event, the
mean training target of its activity — a leakage-free baseline that captures
*where in the process* an event is, which on many logs carries most of the
signal:

.. code-block:: python

   from skpm.baselines import ActivityMeanRegressor

   baseline = ActivityMeanRegressor().fit(train, y_train)

It reads the raw ``activity`` column, so it needs no preprocessing. Inside a
grid search, disable the steps it does not use:

.. code-block:: python

   param_grid = [
       {"model": [LinearRegression(), RandomForestRegressor()]},
       {"features": ["passthrough"], "prefix": ["passthrough"],
        "model": [ActivityMeanRegressor()]},
   ]

A model that cannot beat this is not learning the process.

.. _swapping_estimators:

Other estimators
================

The final step only ever sees rows of numbers, so any estimator with the
scikit-learn interface can take its place. :func:`~sklearn.base.clone` gives a
fresh, unfitted copy of the pipeline and ``set_params`` replaces the named
step, leaving the process-aware half untouched:

.. code-block:: python

   from sklearn.base import clone
   from xgboost import XGBRegressor

   xgb_pipeline = clone(pipeline).set_params(
       model=XGBRegressor(objective="reg:absoluteerror", tree_method="hist")
   )
   xgb_pipeline.fit(train, y_train)

This is the boundary SkPM is built around: SkPM owns everything needing process
semantics, and nothing after that step needs to know it is looking at a
process.

.. minigallery:: skpm.baselines.ActivityMeanRegressor
   :add-heading: Examples
