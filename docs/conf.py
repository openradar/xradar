#!/usr/bin/env python
#
# xradar documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import datetime as dt
import glob
import os
import subprocess
import sys
import warnings
from importlib.metadata import version

sys.path.insert(0, os.path.abspath(".."))

# check readthedocs
on_rtd = os.environ.get("READTHEDOCS") == "True"

# processing on readthedocs
# we need to specifically install the current xradar commit to create version.py
# this fixes the "999" version issue
if on_rtd:
    # install xradar from checked out source
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Installing commit {commit}")
    url = "https://github.com/openradar/xradar.git"
    subprocess.check_call(
        ["python", "-m", "pip", "install", "--no-deps", f"git+{url}@{commit}"]
    )


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx_copybutton",
    "nbsphinx",
]

myst_enable_extensions = [
    "substitution",
]


extlinks = {
    "issue": ("https://github.com/openradar/xradar/issues/%s", "GH"),
    "pull": ("https://github.com/openradar/xradar/pull/%s", "PR"),
}

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?" "config=TeX-AMS-MML_HTMLorMML"
)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "datatree": ("https://xarray-datatree.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "xradar"
copyright = "2022, Open Radar Community"
author = "Open Radar Community"


# get xradar modules and create automodule rst-files
import types

# get xradar version
import xradar  # noqa

modules = []
for k, v in xradar.__dict__.items():
    if isinstance(v, types.ModuleType):
        if k not in ["_warnings", "version"]:
            modules.append(k)
            file = open("{0}.rst".format(k), mode="w")
            file.write(".. automodule:: xradar.{}\n".format(k))
            file.close()

# create API/Library reference rst-file
reference = """
Library Reference
=================

.. toctree::
   :maxdepth: 4
"""

file = open("reference.rst", mode="w")
file.write("{}\n".format(reference))
for mod in sorted(modules):
    file.write("   {}\n".format(mod))
file.close()

# get all rst files, do it manually
rst_files = glob.glob("*.rst")
autosummary_generate = rst_files
autoclass_content = "both"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# get version from metadata
version = version("xradar")
release = version

myst_substitutions = {
    "today": dt.datetime.utcnow().strftime("%Y-%m-%d"),
    "release": release,
}

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "links.rst",
    "**.ipynb_checkpoints",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- nbsphinx specifics --
# always execute notebooks while building docs
nbsphinx_execute = "always"
subprocess.check_call(["cp", "-rp", "../examples/notebooks", "."])

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/xradar_logo.svg"


# Tell the theme where the code lives
# adapted from https://github.com/vispy/vispy
def _custom_edit_url(
    github_user,
    github_repo,
    github_version,
    docpath,
    filename,
    default_edit_page_url_template,
):
    """Create custom 'edit' URLs for API modules since they are dynamically generated."""
    if filename.startswith("generated/"):
        # this is a dynamically generated API page, link to actual Python source
        modpath = os.sep.join(
            os.path.splitext(filename)[0].split("/")[-1].split(".")[:-1]
        )
        if modpath == "modules":
            # main package listing
            modpath = "xradar"
        rel_modpath = os.path.join("..", modpath)
        if os.path.isdir(rel_modpath):
            docpath = modpath + "/"
            filename = "__init__.py"
        elif os.path.isfile(rel_modpath + ".py"):
            docpath = os.path.dirname(modpath)
            filename = os.path.basename(modpath) + ".py"
        else:
            warnings.warn(f"Not sure how to generate the API URL for: {filename}")
    return default_edit_page_url_template.format(
        github_user=github_user,
        github_repo=github_repo,
        github_version=github_version,
        docpath=docpath,
        filename=filename,
    )


html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "openradar",
    "github_repo": "xradar",
    "github_version": "main",
    "doc_path": "docs",
    "edit_page_url_template": "{{ xradar_custom_edit_url(github_user, github_repo, github_version, doc_path, file_name, default_edit_page_url_template) }}",
    "default_edit_page_url_template": "https://github.com/{github_user}/{github_repo}/edit/{github_version}/{docpath}/{filename}",
    "xradar_custom_edit_url": _custom_edit_url,
}


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "announcement": "<p>xradar is in an early stage of development, please report any issues <a href='https://github.com/openradar/xradar/issues'>here!</a></p>",
    "github_url": "https://github.com/openradar/xradar",
    "favicons": [
        {
            "rel": "icon",
            "sizes": "16x16",
            "href": "openradar_micro.svg",
        },
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "openradar_micro.svg",
        },
    ],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xradar",
            "icon": "fas fa-box",
        },
        {
            "type": "local",
            "name": "OpenRadarScience",
            "url": "https://openradarscience.org",
            "icon": "_static/openradar_micro.svg",
        },
    ],
    "navbar_end": ["theme-switcher", "icon-links.html"],
    "use_edit_page_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "xradardoc"

# -- Napoleon settings for docstring processing -------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "scalar": ":term:`scalar`",
    "sequence": ":term:`sequence`",
    "callable": ":py:func:`callable`",
    "file-like": ":term:`file-like <file-like object>`",
    "array-like": ":term:`array-like <array_like>`",
    "Path": "~~pathlib.Path",
}

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
# vlatex_documents = [
#    (master_doc, 'xradar.tex',
#     'xradar Documentation',
#     'Maxwell Grover', 'manual'),
# ]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "xradar", "xradar Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "xradar",
        "xradar Documentation",
        author,
        "xradar",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# make rst_epilog a variable, so you can add other epilog parts to it
rst_epilog = ""
# Read link all targets from file
with open("links.rst") as f:
    rst_epilog += f.read()
