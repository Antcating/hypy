# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

from sphinx.ext.apidoc import main as apidoc_main

ROOT = Path(__file__).resolve().parents[2]  # project root
print(f"Project root determined as: {ROOT}")
sys.path.insert(0, str(ROOT))
print(f"Added {ROOT} to sys.path for Sphinx autodoc")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hypy"
copyright = "2026, Yevhenii Fedorchenko"
author = "Yevhenii Fedorchenko"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": True,
    "titles_only": True,
}


def run_apidoc(app):
    """Generate .rst API files from package modules at build time."""
    docs_source = Path(__file__).resolve().parent
    output_dir = docs_source / "api"
    module_dir = ROOT / "hypy"

    # Remove stale generated files so removed/renamed modules do not linger.
    output_dir.mkdir(parents=True, exist_ok=True)
    for rst_file in output_dir.glob("*.rst"):
        rst_file.unlink()

    apidoc_main(
        [
            "--force",
            "--separate",
            "--module-first",
            "--no-toc",
            "--output-dir",
            str(output_dir),
            str(module_dir),
        ]
    )
