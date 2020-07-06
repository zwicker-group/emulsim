r"""
The `sim` package provides classes for describing physical system that consists
of multiple `elements`, which together describe the state of the system. The
dynamical rules are encoded in `actors`, which either act on individual
elements, encoding their autonomous dynamics, or on multiple elements,
introducing couplings.
"""

__version__ = "0.1"

from .actors import *
from .elements import *
from .simulation import Simulation
from .state import State
from .trackers import *
