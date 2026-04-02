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
    "sphinx.ext.autosummary",
]

autosummary_generate = True

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


def _clean_label(name: str) -> str:
    """Remove long namespace prefixes/suffixes from navigation labels."""
    clean = name
    if clean.startswith("hypy."):
        clean = clean[len("hypy.") :]
    clean = clean.replace("\\_", "_")
    return clean


def _rewrite_apidoc_labels(output_dir: Path) -> None:
    """Rewrite generated apidoc .rst titles and toctree labels for readability."""
    for rst_file in sorted(output_dir.glob("*.rst")):
        text = rst_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines:
            continue

        if rst_file.stem == "modules":
            continue

        title = lines[0].replace(" package", "").replace(" module", "")
        title = _clean_label(title)
        lines[0] = title
        if len(lines) > 1 and set(lines[1]) <= {"=", "-", "~"}:
            lines[1] = "=" * len(title)

        in_toctree = False
        for idx, line in enumerate(lines):
            stripped = line.strip()

            if stripped == ".. toctree::":
                in_toctree = True
                continue

            if in_toctree:
                if not stripped:
                    continue
                if stripped.startswith(":"):
                    continue
                if not line.startswith("   "):
                    in_toctree = False
                    continue
                if "<" in stripped and ">" in stripped:
                    continue

                label = _clean_label(stripped)
                lines[idx] = f"   {label} <{stripped}>"

        rst_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_apidoc(app):
    """Generate .rst API files from package modules at build time."""
    docs_source = Path(__file__).resolve().parent
    output_dir = docs_source / "api"
    module_dir = ROOT / "hypy"

    apidoc_main(
        [
            "--force",
            "--separate",
            "--module-first",
            "--output-dir",
            str(output_dir),
            str(module_dir),
        ]
    )
    _rewrite_apidoc_labels(output_dir)


def skip_external_members(app, what, name, obj, skip, options):
    """Skip documenting members not defined in the hypy namespace."""
    if skip:
        return True

    module_name = getattr(obj, "__module__", "")
    if module_name and not module_name.startswith("hypy"):
        return True

    return None


def setup(app):
    app.connect("builder-inited", run_apidoc)
    app.connect("autodoc-skip-member", skip_external_members)
