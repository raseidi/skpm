"""Gallery ordering for sphinx-gallery.

This lives in a module rather than in ``conf.py`` on purpose. Sphinx pickles its
build environment, and a class or function defined in ``conf.py`` is not
picklable — the build dies at ``pickling environment`` with
``Can't pickle <function ...>: attribute lookup ... on __main__ failed``.
sphinx-gallery resolves ``within_subsection_order`` from a dotted string, so
pointing it at this module keeps only a string in the config.
"""

import os

from sphinx_gallery.sorting import FileNameSortKey

#: Teaching order for the gallery: start with the end-to-end workflow, then the
#: task-specific pages, then reference material. Anything not listed sorts after
#: these, by filename.
EXAMPLE_ORDER = (
    "plot_quickstart.py",
    "plot_next_activity.py",
    "plot_prefix_encoding.py",
    "plot_process_features.py",
    "plot_download_event_logs.py",
    "bpi20_remaining_time.py",
)


class ExampleOrder(FileNameSortKey):
    """Sort examples into :data:`EXAMPLE_ORDER`, not alphabetically."""

    def __call__(self, filename):
        # sphinx-gallery passes a str on some code paths and a Path on others.
        name = os.path.basename(str(filename))
        rank = (
            EXAMPLE_ORDER.index(name)
            if name in EXAMPLE_ORDER
            else len(EXAMPLE_ORDER)
        )
        return f"{rank:03d}{name}"
