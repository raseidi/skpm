.. Title and tagline are real reStructuredText, not raw HTML, so Sphinx knows
   the document's title — it drives the browser tab, the breadcrumb and the
   sidebar. The hero look is CSS, scoped to the tagline's class.

SkPM
====

.. rst-class:: skpm-hero-tagline

Process mining, the scikit-learn way.

Event logs are not tables of independent rows. Cases unfold over time, every
event is a moment where a prediction could be made, and almost every convenient
shortcut leaks the future into the past.

**SkPM turns an event log into a supervised learning problem without leaking it.**
Feature extraction, prefix encoding, prediction targets and time-aware splits —
all ordinary scikit-learn transformers, so they compose with the ``Pipeline``,
``ColumnTransformer`` and ``GridSearchCV`` you already use.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: skpm-cta

   .. grid-item::

      .. button-ref:: auto_examples/plot_quickstart
         :ref-type: doc
         :color: primary
         :expand:

         Quickstart

   .. grid-item::

      .. button-ref:: user_guide/index
         :ref-type: doc
         :color: secondary
         :outline:
         :expand:

         Read the User Guide


From event log to prediction
----------------------------

Download a public benchmark log, split it without leaking, build a target, and
fit a model.

.. code-block:: python

   from sklearn.ensemble import HistGradientBoostingRegressor
   from sklearn.pipeline import Pipeline

   from skpm.event_logs import BPI17
   from skpm.feature_extraction import TimestampExtractor
   from skpm.feature_extraction.targets import remaining_time
   from skpm.model_selection import train_test_split
   from skpm.sequence_encoding import Aggregation

   # Split first — before extracting features, before fitting anything.
   train, test = train_test_split(BPI17(), strategy="unbiased")

   # The target is a modelling choice, so you make it. One value per event.
   y_train = remaining_time(train, time_unit="h")

   Pipeline([
       ("time", TimestampExtractor(time_unit="h")),
       ("prefix", Aggregation(method="mean")),
       ("model", HistGradientBoostingRegressor()),
   ]).fit(train, y_train)

Your own log works the same way — ``train_test_split(pd.read_csv("log.csv"))``.
No loader class, no conversion step to remember.


Why SkPM
--------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`shield-check` Leakage is the default enemy

      Remaining time and next activity are *targets*, so they live outside the
      feature pipeline. ``TimestampExtractor`` exposes past-looking features
      only — a future-looking one cannot be selected by accident. Splits keep
      whole cases on one side, and the ``unbiased`` strategy drops the cases
      that the recording window truncates.

   .. grid-item-card:: :octicon:`plug` Nothing new to learn

      ``fit`` / ``transform`` / ``predict``, exactly as you know them. Because
      SkPM transformers are scikit-learn transformers, cross-validation, grid
      search and every metric come for free rather than being reimplemented.

   .. grid-item-card:: :octicon:`database` Public logs, one line

      Fourteen BPI Challenge and Sepsis logs download and cache themselves on
      first use. The six that have published unbiased-split parameters carry
      them, so ``strategy="unbiased"`` is reproducible without copying constants
      out of a paper.

   .. grid-item-card:: :octicon:`git-merge` One shape, everywhere

      Every log in SkPM is a DataFrame indexed by ``(case_id, timestamp,
      event_id)``. Case and time are identity, so they survive every pipeline
      step — you can still ask *which case, and when* of a prediction made ten
      transformations later.


What is in the box
------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Feature extraction
      :link: user_guide/feature_extraction
      :link-type: doc

      Temporal features, resource pools discovered from handover behaviour, and
      inter-case work-in-progress.

   .. grid-item-card:: Sequence encoding
      :link: user_guide/sequence_encoding
      :link-type: doc

      Turn a growing prefix into a fixed-length vector — aggregate it, index it,
      or slide a window over it.

   .. grid-item-card:: Targets and splits
      :link: user_guide/model_selection
      :link-type: doc

      Remaining time, next activity and execution time; temporal and unbiased
      splitting.

   .. grid-item-card:: Event logs
      :link: user_guide/event_logs
      :link-type: doc

      XES and CSV parsing, column-name normalization, and the public benchmark
      loaders.


Install
-------

.. code-block:: bash

   pip install skpm

Python 3.11 or newer. A PyPI release is on the way; until then, install from the
repository.


Where to start
--------------

* :doc:`Quickstart <auto_examples/plot_quickstart>` — the whole workflow on one
  page, running on synthetic data in seconds.
* :doc:`User Guide <user_guide/index>` — each step explained once, in order:
  the event log, splitting, targets, features, prefix encoding, composition.
* :doc:`Examples <auto_examples/index>` — next-activity prediction, the four
  prefix encodings side by side, process-specific features, and the full
  workflow on a real BPI Challenge log.
* :doc:`API reference <autoapi/index>` — every public class and function.

.. The toctrees below are hidden: they build the sidebar and the page order
   without repeating the links already written above.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   autoapi/index


Citing SkPM
-----------

SkPM was presented at the CoopIS 2023 demonstration track.

.. code-block:: bibtex

   @inproceedings{OyamadaTJC23,
     author    = {Rafael Seidi Oyamada and
                  Gabriel Marques Tavares and
                  Sylvio Barbon Junior and
                  Paolo Ceravolo},
     title     = {A Scikit-learn Extension Dedicated to Process Mining Purposes},
     booktitle = {Proceedings of the Demonstration Track co-located with the
                  International Conference on Cooperative Information Systems 2023},
     series    = {{CEUR} Workshop Proceedings},
     publisher = {CEUR-WS.org},
     year      = {2023},
   }

Contributions are welcome — see the
`repository <https://github.com/raseidi/skpm>`_ for issues and guidelines.
