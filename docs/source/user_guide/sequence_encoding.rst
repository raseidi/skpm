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

Only :class:`Aggregation` requires numeric input — averaging a string activity
has no meaning, so it raises. The positional encoders shift values of any dtype,
and :class:`Bucketing` ignores them entirely. Even so, encode categorical
columns *before* the encoder rather than after; see :ref:`encoding_order`.

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
   downstream.

   ``fill_value`` is applied as-is to every column, so on a **string** column
   the default pads with the integer ``0`` and leaves a mixed-type object
   column that scikit-learn's encoders reject outright::

       TypeError: Encoders require their input argument must be uniformly
       strings or numbers. Got ['int', 'str']

   Pass a string sentinel — ``fill_value="<pad>"`` — whenever you encode a
   categorical column directly.

.. _encoding_order:

Encode categories before, not after
===================================

For a categorical column such as ``activity``, one-hot encode it *before* the
prefix encoder. The two orders are not equivalent, and the difference is not a
matter of taste.

:class:`~sklearn.preprocessing.OneHotEncoder` learns an **independent vocabulary
per column**. Encode after windowing and each position gets its own, discovered
from whichever values happened to land there in training:

.. code-block:: text

   position 0 vocabulary: ['pick', 'receive', 'ship']
   position 1 vocabulary: ['<pad>', 'pick', 'receive']    <- 'ship' missing

Here ``ship`` never occurred as a *previous* event during fit, so at prediction
time "the previous event was ship" has no column to land in and silently
collapses to all zeros. The output width also varies from position to position.

Encoding first avoids both: one encoder, one vocabulary, a column space that is
identical at fit and transform, and ``handle_unknown="ignore"`` handling drift
in one place. It is also what makes :class:`Aggregation` meaningful, since the
mean of a one-hot column is a relative frequency.

.. warning::

   One consequence is worth knowing. With one-hot encoding first, an absent
   window position is an all-zeros block — and ``handle_unknown="ignore"``
   produces *exactly the same* all-zeros block for a category unseen at fit.
   "No previous event" and "previous event was an activity I have never seen"
   become indistinguishable.

   A real one-hot block sums to 1, so the distinction survives in principle,
   but a tree model cannot compute a row-sum across columns. If it matters for
   your task, add :func:`skpm.trace_positions` as a feature so the model can
   condition on how many positions are genuinely filled.

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
