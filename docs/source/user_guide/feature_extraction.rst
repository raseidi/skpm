.. _feature_extraction:

===================
Feature extraction
===================

.. currentmodule:: skpm.feature_extraction

Feature extraction turns the log's columns and timestamps into numbers. All the
transformers here are scikit-learn transformers: they implement
``fit``/``transform``, they compose in a ``Pipeline``, and — with the exception
noted in :ref:`case_level_features` — they return one row per event, carrying
the event-log index unchanged.

.. _timestamp_features:

Temporal features
=================

:class:`TimestampExtractor` derives numeric features from the timestamps. They
come at two levels, and the distinction matters:

* **case-level** features describe the event's position *in its own case*;
* **event-level** features describe the moment itself, independent of any case.

.. code-block:: python

   from skpm.feature_extraction import TimestampExtractor

   TimestampExtractor(
       case_features=["accumulated_time", "time_since_last_event"],
       event_features=["hour_of_day", "day_of_week"],
       time_unit="h",
   ).fit_transform(log)

Each level has a fixed registry of feature names. ``"all"`` selects exactly the
registry, a list selects a subset, ``None`` selects none, and an unknown name
raises rather than being ignored.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Level
     - Features
   * - ``case_features``
     - ``accumulated_time`` (elapsed since the case started),
       ``time_since_last_event``
   * - ``event_features``
     - ``day_of_month``, ``day_of_week``, ``day_of_year``, ``hour_of_day``,
       ``min_of_hour``, ``month_of_year``, ``numerical_timestamp``,
       ``sec_of_min``, ``secs_since_sunday``, ``secs_within_day``,
       ``week_of_year``

``time_unit`` (``"s"``, ``"m"``, ``"h"``, ``"d"``, ``"w"``) applies to the
duration features. The cyclical event-level features are encoded to roughly
±0.5 instead of their raw calendar value, so 9 a.m. does not read as ``9`` and
midnight is not maximally distant from 23:59.

Note what is *not* in the registry. Only past-looking features are exposed:
``remaining_time`` and ``execution_time`` are marked target-only and cannot be
selected as inputs. To predict them, build them with
:mod:`skpm.feature_extraction.targets` and pass them as ``y`` (see
:ref:`targets`).

.. _inter_case_features:

Inter-case features
===================

Everything above looks at one case at a time. :class:`WorkInProgress` looks
across them: it partitions time into consecutive windows and labels each event
with the number of distinct cases active in its window.

.. code-block:: python

   from skpm.feature_extraction import WorkInProgress

   WorkInProgress(window_size="1D").fit_transform(log)   # column: "wip"

``window_size`` is any pandas offset alias — ``"1D"``, ``"12h"``, ``"1W"``.
This is a genuine process signal: a case queued behind two hundred others
behaves differently from the same case in a quiet week, and no per-case feature
can see that.

.. warning::

   ``WorkInProgress`` is the one transformer here that reads other cases. Fit
   it on the training log only — inside the ``Pipeline``, never on the full log
   beforehand — or the count for a training event can reflect test-set cases.

.. _resource_features:

Resource roles
==============

:class:`ResourcePoolExtractor` discovers organisational roles from behaviour
rather than from an org chart. It correlates each resource's activity profile
with every other's, links the pairs above ``threshold``, and assigns each
resource the connected component it falls in [1]_.

.. code-block:: python

   from skpm.feature_extraction import ResourcePoolExtractor

   ResourcePoolExtractor(threshold=0.7).fit_transform(log)  # column: "resource_roles"

Two resources who perform the same mix of activities land in the same role even
if they never appear in the same case. This needs the optional ``resource``
column (see :ref:`column_naming`); it is the reason that field exists.

A resource unseen at fit time is mapped to ``UNK`` and raises a
:class:`~skpm.warnings.ConceptDriftWarning` — new people joining the process is
normal, but worth knowing about rather than silently absorbing.

.. _case_level_features:

Case-level features
===================

:class:`~skpm.feature_extraction.case.variant.VariantExtractor` describes whole
cases, not events. A *trace variant* is the ordered tuple of activities of a
case; cases sharing one followed the same path.

.. code-block:: python

   from skpm.feature_extraction.case.variant import VariantExtractor

   variants = VariantExtractor().fit_transform(log)   # one row per case

Because it emits one row per **case**, it does not align with an event-level
target and is a **terminal** step, not a pipeline intermediate. Use it for
analysis — how many distinct paths exist, how concentrated they are, which rare
variants are worth inspecting — rather than as an ``X`` for an event-level
model. ``inverse_transform`` maps the integer codes back to activity tuples.

.. topic:: References

   .. [1] Minseok Song and Wil M. P. van der Aalst, "Towards comprehensive
      support for organizational mining", *Decision Support Systems*, 2008.

.. minigallery:: skpm.feature_extraction.TimestampExtractor skpm.feature_extraction.WorkInProgress skpm.feature_extraction.ResourcePoolExtractor skpm.feature_extraction.case.VariantExtractor
   :add-heading: Examples
