r"""
The `sim` package provides classes for describing physical system that consists of
multiple `elements`, which together describe the state of the system. The dynamical
rules are encoded in `actors`, which either act on individual elements, encoding their
autonomous dynamics, or on multiple elements, introducing couplings.
"""

# determine the package version
try:
    # try reading version from the automatically generated module
    from ._version import __version__  # type: ignore
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("sim")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine any version
        __version__ = "unknown"
    del PackageNotFoundError, version  # clean name space

# make key classes from modelrunner available
from modelrunner import Parameter

# import key classes from py-sim package into general namespace
from .actors import *
from .elements import *
from .simulation import Simulation
from .state import State
from .trackers import *
