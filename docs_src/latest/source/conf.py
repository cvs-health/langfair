# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../langfair")
)  # lets sphinx find llambda code

project = "LangFair"
copyright = "2024, CVS Health"
author = "Dylan Bouchard"
version = importlib.metadata.version("langfair")
release = ".".join(version.rsplit(".")[:-1])

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

bibtex_bibfiles = ["refs.bib"]

autodoc_mock_imports = ["sentence_transformers", "transformers"]

autosummary_generate = True

templates_path = ["_templates"]

html_static_path = ["_static"]

exclude_patterns = []

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/cvs-health/langfair",
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {
        "image_light": "_static/images/langfair-logo.png",
        "image_dark": "_static/images/langfair-logo2.png",
    },
}

html_favicon = "_static/images/langfair-logo-only.png"

sphinx_gallery_conf = {
    # "reference_url": {"langfair": None},
    "examples_dirs": "../../../example_scripts",
    "gallery_dirs": "auto_examples",
    # pypandoc enables rst to md conversion in downloadable notebooks
    # "pypandoc": True,
    # "doc_module": ("langfair",),
}

source_suffix = [".rst"]
