# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from sphinx_gallery.sorting import FileNameSortKey
from pathlib import Path

# find project
sys.path.insert(0, str(Path(__file__).parents[1]))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skpm"
copyright = "2024, Rafael Oyamada"
author = "Rafael Oyamada"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

autoapi_dirs = ["../src"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

sg_examples_dir = "../examples"
sg_gallery_dir = "auto_examples"
sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": [sg_examples_dir],
    # path to where to save gallery generated output
    "gallery_dirs": [sg_gallery_dir],
    # specify that examples should be ordered according to filename
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("SampleModule"),
    "filename_pattern": "/*.py",
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/{.major}".format(sys.version_info),
        None,
    ),
    "matplotlib": ("https://matplotlib.org/", None),
}
