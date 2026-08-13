from importlib.metadata import version as _version

project = "skpm"
copyright = "2026, Rafael S. Oyamada"
author = "Rafael S. Oyamada"
release = _version("skpm")

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    # Only plot_*.py EXECUTES. Heavy BPI examples render as code, unrun.
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
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
