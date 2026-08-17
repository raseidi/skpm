.. _event_logs:

==============
The event log
==============

.. currentmodule:: skpm

Every transformer, target and split in SkPM agrees on one representation of an
event log. Learning it once is enough; nothing else in the package asks you to
reshape your data again.

.. _event_log_shape:

The canonical shape
===================

An event log is a :class:`pandas.DataFrame` in which each row is a single
event, indexed by a three-level :class:`pandas.MultiIndex`:

===============  ============================================================
Level            Meaning
===============  ============================================================
``case_id``      The process instance the event belongs to.
``timestamp``    When the event happened, in UTC.
``event_id``     A 0-based counter over the whole log, assigned after a
                 stable sort by ``(case_id, timestamp)``.
===============  ============================================================

Everything else — ``activity``, ``resource``, and any attribute of your own —
stays an ordinary column.

That division is the one design decision to internalise. **Columns are data:**
feature extraction consumes them and replaces them, so a column is gone by the
end of a pipeline. **Index levels are identity:** they are never consumed, so
they survive every step. Case and time are identity, which is why a prediction
made ten transformations downstream can still be traced back to *which case,
and when*.

``event_id`` exists because ``(case_id, timestamp)`` is not unique: a case can
record two events at the same instant. The counter keeps such events
distinguishable, in their original document order.

.. _to_event_log:

Building it with ``to_event_log``
=================================

:func:`~skpm.to_event_log` is the single conversion boundary in SkPM. It takes
a flat DataFrame and returns the canonical shape:

.. code-block:: python

   import pandas as pd
   from skpm import to_event_log

   raw = pd.DataFrame(
       {
           "case:concept:name": ["PO-2", "PO-1", "PO-1", "PO-1"],
           "time:timestamp": [
               "2024-02-01 11:00+01:00",
               "2024-02-01 09:00+01:00",
               "2024-02-01 10:00+01:00",
               "2024-02-01 10:00+01:00",
           ],
           "concept:name": ["ship", "receive", "approve", "release"],
       }
   )
   log = to_event_log(raw)

In one call it normalizes the column names, parses the timestamps to UTC, sorts
stably by ``(case_id, timestamp)``, numbers the events, and moves ``case_id``
and ``timestamp`` into the index::

                                              activity
   case_id timestamp                 event_id
   PO-1    2024-02-01 08:00:00+00:00 0          receive
           2024-02-01 09:00:00+00:00 1          approve
                                     2          release
   PO-2    2024-02-01 10:00:00+00:00 3             ship

Note the two ``PO-1`` events at ``09:00``: identical case and timestamp,
different ``event_id``, source order preserved.

Calling ``to_event_log`` on an already-canonical log returns it unchanged, so
it is safe to apply defensively.

.. _column_naming:

Naming your columns
===================

SkPM expects **one** source name per field, defaulting to the XES standard —
``case:concept:name``, ``time:timestamp``, ``concept:name`` and the optional
``org:resource``. There is deliberately no alias guessing: a column named
``CaseID`` is not silently interpreted as a case identifier, because guessing
wrongly is worse than asking.

Declare your naming per call with ``column_mapping``, whose keys are SkPM
meanings and whose values are your column names:

.. code-block:: python

   log = to_event_log(
       raw,
       column_mapping={
           "case_id": "Order",
           "timestamp": "Occurred",
           "activity": "Step",
           "resource": "Operator",
       },
   )

Or declare it once for the session, if all your logs are named the same way:

.. code-block:: python

   from skpm.config import EventLogConfig

   EventLogConfig.set_global_config(case_id="Order", timestamp="Occurred",
                                    activity="Step")
   EventLogConfig.reset_global_config()  # back to the XES defaults

``case_id``, ``timestamp`` and ``activity`` are required and raise if
unresolved; ``resource`` is optional, and only two transformers need it.
Columns you do not map — business attributes such as an order amount — pass
through untouched.

The *canonical* names are constants, not configuration. What you configure is
only how your **source** columns are spelled.

.. _index_accessors:

Reading the index
=================

Four accessors pull an index level out as a :class:`pandas.Series` aligned to
the log. Prefer them to ``log.index.get_level_values(...)``:

=================================  ==========================================
Accessor                           Returns
=================================  ==========================================
:func:`~skpm.case_ids`             The case each event belongs to.
:func:`~skpm.timestamps`           When each event happened.
:func:`~skpm.event_ids`            The global event counter.
:func:`~skpm.trace_positions`      0-based position *within* the case.
=================================  ==========================================

``trace_positions`` is the one that is not an index level: it is derived, and
it is what you want for "how far into its case is this event?" — plotting error
against prefix length, for instance, or grouping predictions by how much
history they had.

.. code-block:: python

   from skpm import case_ids, trace_positions

   n_cases = case_ids(log).nunique()
   depth = trace_positions(log)

Because features and targets both carry this index, they align with no merge
and no identifier columns::

   X = TimestampExtractor().fit_transform(log)
   y = next_activity(log)
   X.index.equals(y.index)   # True

.. _log_like:

What counts as a log
====================

Every SkPM API that takes a log accepts three forms, named collectively by the
:data:`~skpm.LogLike` alias:

* an :class:`~skpm.event_logs.base.EventLog` — any loader in
  :mod:`skpm.event_logs`;
* a flat :class:`pandas.DataFrame`;
* an already-canonical DataFrame.

They all route through :func:`~skpm.to_event_log`, so ``.dataframe`` is
available on a loader but never required:

.. code-block:: python

   TimestampExtractor().fit(BPI17())          # EventLog
   remaining_time(BPI17())                    # same object, different API
   train_test_split(pd.read_csv("mine.csv"))  # flat DataFrame

.. warning::

   The convenience stops at scikit-learn's door, in two places.

   **Plain estimators need a feature matrix.**
   ``RandomForestRegressor().fit(BPI17())`` fails by design: a raw log has
   string activities and nothing encoded. Put a SkPM transformer in front of
   it, which is what a ``Pipeline`` does.

   **Meta-estimators need a DataFrame.** :class:`~sklearn.model_selection.GridSearchCV`,
   :func:`~sklearn.model_selection.cross_val_score` and friends call
   ``indexable(X, y, groups)`` before any SkPM code runs, and an ``EventLog``
   has no ``iloc``, ``__len__`` or ``shape``. Cross-validation therefore needs
   a DataFrame — which is what you are holding anyway, since
   :func:`~skpm.model_selection.train_test_split` returns two of them.

   Treat ``EventLog`` input as a convenience for exploratory one-liners, not
   the headline workflow.

.. _public_logs:

Public event logs
=================

:mod:`skpm.event_logs` ships loaders for fourteen public datasets — the BPI
Challenge logs and Sepsis. Constructing one downloads it from the 4TU
repository on first use, parses it, normalizes the columns, and caches it
locally; later constructions read the cache.

.. code-block:: python

   from skpm.event_logs import BPI13OpenProblems

   log = BPI13OpenProblems()
   print(log)                # name, number of cases and events
   print(log.cache_folder)   # where it was stored

Six of the fourteen also publish the parameters for the unbiased split; see
:ref:`unbiased_split`.

To read a XES file you already have, use :func:`~skpm.event_logs.read_xes`,
which returns a flat DataFrame for :func:`~skpm.to_event_log`:

.. code-block:: python

   from skpm import to_event_log
   from skpm.event_logs import read_xes

   log = to_event_log(read_xes("my_log.xes", n_jobs=-1))

Anything pandas can read works the same way — CSV, Parquet, a database query.

.. minigallery:: skpm.to_event_log skpm.event_logs.BPI13OpenProblems skpm.trace_positions
   :add-heading: Examples
