.. _sequence_encoding:

==================
Sequence encoding
==================

.. currentmodule:: skpm.sequence_encoding

.. _prefixes:

Every event is a prediction moment
==================================

In predictive process monitoring you do not predict once per case. You predict
at *every* event, using only what has happened up to and including it. That
growing slice is the case's **prefix**.

A case with six events therefore contributes six training rows: the prefix of
length 1, of length 2, and so on. This is why SkPM keeps one row per event, and
why the targets are also one value per event.

Sequence encoding is what turns a variable-length prefix into the fixed-length
vector a tabular model needs. It is **optional**: without it, each event is an
independent sample described only by its own features — a reasonable model, and
the right starting point. Add an encoder when you want the model to see the
path taken, not just the current step.

All four encoders below take numeric input, so one-hot your categorical columns
first (see :ref:`composing`).

.. _aggregation:

``Aggregation`` — order-agnostic
================================

:class:`Aggregation` summarises the prefix with a running statistic per column,
discarding order:

.. code-block:: python

   from skpm.sequence_encoding import Aggregation

   Aggregation(method="mean").fit_transform(X)

``method`` is ``"mean"``, ``"sum"``, ``"median"`` or ``"norm"``. The output
keeps the input's column names and width.

The mean of a one-hot activity column is the **relative frequency** of that
activity so far — a compact summary of the path taken. That single fact is what
makes this the usual default.

``prefix_len`` bounds the window; left at ``None`` the aggregation is
cumulative over the whole prefix.

.. _windowing:

``Windowing`` — the recent past, in order
=========================================

:class:`Windowing` keeps order instead of collapsing it, giving a fixed-width
view of the ``n`` most recent events:

.. code-block:: python

   from skpm.sequence_encoding import Windowing

   Windowing(n=3).fit_transform(X)

Column ``<attr>_w_0`` is the current event, ``_w_1`` the one before it, and so
on — positions *relative* to the current event. The first events of a case have
no ``_w_1``, and those cells are padded.

.. _indexing:

``Indexing`` — absolute positions
=================================

:class:`Indexing` encodes each position from the start of the case:
``<attr>_pos_0`` is the case's first event, ``_pos_1`` its second. A value is
revealed only once the prefix has reached it, so a prefix never sees an event
beyond its own length.

.. code-block:: python

   from skpm.sequence_encoding import Indexing

   Indexing().fit_transform(X)

The width is the longest case seen at fit, so the whole prefix stays visible.
There is deliberately no ``n``: a fixed width shorter than a case would freeze
every longer prefix onto its first ``n`` events, collapsing them to one
identical vector.

.. note::

   Both positional encoders pad structurally-missing cells with ``fill_value``
   (``0`` by default), so the output is NaN-free and usable by estimators that
   reject missing values. Pass ``fill_value=None`` to keep ``NaN`` and impute
   downstream, or a string such as ``"<none>"`` for categorical columns.

.. _bucketing:

``Bucketing`` — grouping prefixes
=================================

:class:`Bucketing` does not encode a prefix; it labels one, so you can train a
separate model per group:

.. code-block:: python

   from skpm.sequence_encoding import Bucketing

   Bucketing(method="prefix").fit_transform(log)   # column: "bucket"

``method="single"`` puts every event in one bucket (the no-op baseline);
``method="prefix"`` buckets by trace position, so all first events share a
bucket, all second events another. Predicting from a two-event prefix and from
a twenty-event prefix are close to different problems, and bucketing is the
classical way to admit that.

.. _choosing_an_encoder:

Choosing one
============

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Encoder
     - Keeps order?
     - Use when
   * - :class:`Aggregation`
     - No
     - You want *what* happened, not *when*. Compact, fixed width, the usual
       default.
   * - :class:`Windowing`
     - Yes, relative
     - Recent history drives the outcome, e.g. next-activity prediction.
   * - :class:`Indexing`
     - Yes, absolute
     - Position from the case start is meaningful and cases are short. Widest
       output.
   * - :class:`Bucketing`
     - n/a
     - You intend to fit a separate model per prefix group.

The three are not exclusive: a ``FeatureUnion`` of :class:`Aggregation` and
:class:`Windowing` gives a model both the summary and the recent order.

.. minigallery:: skpm.sequence_encoding.Aggregation skpm.sequence_encoding.Windowing skpm.sequence_encoding.Indexing skpm.sequence_encoding.Bucketing
   :add-heading: Examples
