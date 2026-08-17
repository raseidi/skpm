import sys
from importlib.metadata import version as _version
from pathlib import Path

# So sphinx-gallery can import `_gallery_order` by name (see below).
sys.path.insert(0, str(Path(__file__).parent))

project = "skpm"
copyright = "2026, Rafael S. Oyamada"
author = "Rafael S. Oyamada"
release = _version("skpm")

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# The loader docstrings cite their dataset as :doi:`Title <https://doi.org/...>`.
# Without a registered role that is an "Unknown interpreted text role" error on
# every loader's API page. The explicit-title form fills %s with the target, so
# a bare "%s" pattern makes the role a plain external link.
extlinks = {"doi": ("%s", None)}

# sphinx-gallery creates this while generating the gallery, but autoapi fires
# `autodoc-process-docstring` first and sphinx-gallery's handler writes a stub
# into it on the spot. Create it here so the build does not depend on the order
# the two extensions are listed in.
_BACKREFERENCES_DIR = "gen_modules/backreferences"
(Path(__file__).parent / _BACKREFERENCES_DIR).mkdir(parents=True, exist_ok=True)

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    # Teaching order, not sphinx-gallery's default. Must be a dotted string,
    # not a callable — see docs/source/_gallery_order.py.
    "within_subsection_order": "_gallery_order.ExampleOrder",
    # Only plot_*.py EXECUTES. Heavy BPI examples render as code, unrun.
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    # Record which skpm objects each example uses, so the user guide can end a
    # section with `.. minigallery:: skpm.sequence_encoding.Aggregation`. The
    # stubs are written from the example *code*, not maintained by hand.
    #
    # Only the *manual* directive is useful here. sphinx-gallery's automatic
    # injection into API pages keys off `autodoc-process-docstring`; autoapi
    # does emit that event, but with unqualified names (`Aggregation`, not
    # `skpm.sequence_encoding.Aggregation`), so the stubs it looks for never
    # match the ones written from the examples.
    "doc_module": ("skpm",),
    "backreferences_dir": _BACKREFERENCES_DIR,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# --- API reference (static parsing; never imports skpm) ---
autoapi_dirs = ["../../src/skpm"]
autoapi_type = "python"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
