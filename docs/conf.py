"""Sphinx config."""

import os

project = "Shabadoo"

extensions = [
    "m2r",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = []
exclude_patterns = []

html_theme = "classic"
html_static_path = []
html_show_copyright = False
html_show_sphinx = False

if os.getenv("SPHINX_BUILD_PROD"):
    html_baseurl = "https://nolanbconaway.github.io/shabadoo/"
    templates_path.append("_templates")
