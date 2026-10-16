# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

import sphinx_rtd_theme

# Prefer importing clarans from the installed environment (with compiled Cython extensions).
# Only fallback to inserting the project root if it is not installed.
try:
    import clarans
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import clarans

project = "scikit-clarans"
copyright = "2026, Nguyễn Ngọc Thiện"
author = "Nguyễn Ngọc Thiện"

version = release = clarans.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

from sphinx_gallery.sorting import ExplicitOrder

# Sphinx-Gallery settings (scikit-learn standard)
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "subsection_order": ExplicitOrder(
        [
            "../../examples/cluster_geometry",
            "../../examples/distance_metrics",
            "../../examples/initialization",
            "../../examples/inference",
            "../../examples/robustness",
            "../../examples/benchmarks",
        ]
    ),
    "filename_pattern": r"plot_",
    "ignore_pattern": r"__init__\.py",
    "download_all_examples": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib",),
    "show_memory": False,
    "plot_gallery": "True",
    "recommender": {
        "enable": True,
        "n_examples": 4,
    },
}

# Copybutton settings: strip Python/terminal prompts when copying
copybutton_prompt_text = r"^>>> |^\.\.\. |^\$ "
copybutton_prompt_is_regexp = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx mapping to resolve references to external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": True,
}
html_static_path = ["_static"]

html_css_files = [
    "gallery.css",
    "github-link.css",
]

html_js_files = [
    "gallery.js",
    "github-link.js",
]

html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}


def remove_gallery_nested_toctree(app, docname, source):
    """Remove redundant toctree appended at the end of auto_examples/index.rst.

    Sphinx-Gallery appends a toctree of all subsection indices at the end of
    auto_examples/index.rst. In docutils AST, this places the toctree inside
    the last section ('Benchmarks and Convergence'), causing the sidebar to
    display all categories nested under the benchmarks section.
    Since all subsection indices are already included in docs/source/index.rst,
    we strip this trailing toctree so the gallery index displays clean headings.
    """
    if docname == "auto_examples/index":
        import re

        source[0] = re.sub(
            r"\.\. toctree::\s+:hidden:\s+:includehidden:.*?(?=\.\. only::|\Z)",
            "",
            source[0],
            flags=re.DOTALL,
        )


def setup(app):
    app.connect("source-read", remove_gallery_nested_toctree)
