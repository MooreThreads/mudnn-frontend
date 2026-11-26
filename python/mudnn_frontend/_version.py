"""Single source of truth for the frontend version.

``pyproject.toml`` reads it through ``[tool.setuptools.dynamic]``, and the
package re-exports it as ``mudnn_frontend.__version__`` / ``mudnn.__version__``.
A bump is therefore one edit here instead of one per shipped location.
"""

__version__ = "0.1.0"
