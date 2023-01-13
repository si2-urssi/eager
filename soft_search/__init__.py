"""Top-level package for soft_search."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soft-search")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamxb@uw.edu"
