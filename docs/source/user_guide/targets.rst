.. _targets:

===================
Prediction targets
===================

.. currentmodule:: skpm.feature_extraction.targets

A target is what you are trying to predict — ``y`` in
``estimator.fit(X, y)``. SkPM provides three, covering the two standard
predictive process monitoring tasks.

.. _available_targets:

The three targets
=================

Each is a **function** taking a log and returning a 1-D
:class:`pandas.Series` aligned to the event-log index — one value per event,
not one per case.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Target
     - Task
     - Value at each event
   * - :func:`remaining_time`
     - regression
     - Time until its case finishes. Reaches ``0`` at the last event.
   * - :func:`execution_time`
     - regression
     - Time until the *next* event of the case — the next-event-time task.
       ``0`` at the last event.
   * - :func:`next_activity`
     - classification
     - The activity that follows, or ``<EOT>`` at the last event.

.. code-block:: python

   from skpm.feature_extraction.targets import (
       execution_time, next_activity, remaining_time,
   )

   y_train = remaining_time(train, time_unit="h")
   y_test = remaining_time(test, time_unit="h")

``time_unit`` is one of ``"s"``, ``"m"``, ``"h"``, ``"d"``, ``"w"`` and
defaults to seconds. Choose it once and report your error in the same unit;
a mean absolute error of ``172800`` is a fact about seconds, not about your
model.

Build the target **once per side**, after splitting. Switching tasks is then a
two-line change — pass ``next_activity(train)`` as ``y`` and put a classifier
at the end of the pipeline.

.. _targets_not_features:

Why they are functions, not transformers
========================================

All three read *future* events. That is exactly what makes them targets, and
exactly why they must never enter the feature matrix: a pipeline step computing
remaining time would hand the model the answer.

Keeping them as functions makes that structural rather than advisory. They
cannot be added to a ``Pipeline`` by accident, which follows scikit-learn's own
separation — ``y`` is passed alongside the pipeline, never through it.

The same reasoning constrains :class:`~skpm.feature_extraction.TimestampExtractor`:
its ``remaining_time`` and ``execution_time`` computations are marked
target-only and cannot be selected as input features. Ask for one and it
raises. See :ref:`timestamp_features`.

.. _target_alignment:

Alignment
=========

Because a target carries the event-log index, it lines up with any feature
matrix built from the same log without merging, sorting or an identifier
column:

.. code-block:: python

   X = TimestampExtractor().fit_transform(train)
   y = remaining_time(train, time_unit="h")

   X.index.equals(y.index)   # True

It also survives ``groupby``, which is what makes post-hoc analysis cheap —
error by case, by day, or by how far into a case the prediction was made:

.. code-block:: python

   from skpm import trace_positions

   errors = (y_test - predictions).abs()
   errors.groupby(trace_positions(test)).mean()

.. minigallery:: skpm.feature_extraction.targets.remaining_time skpm.feature_extraction.targets.next_activity skpm.feature_extraction.targets.execution_time
   :add-heading: Examples
