r"""
The `sim` package provides classes and methods for simulating the dynamics
of emulsions using an agent-based model.
In this model, several agents interact with a common background, which defines
the space in which agents can interact.
The agents define how material is exchanged with the background and the
background defines how material is distributed.

To describe the dynamics of emulsions, we describe droplets by their positions
and radii, while the dilute phase (the background) is a continuous field with
little variations.
The dynamics in the background can thus be described by a coarsely discretized
diffusion equation.
Conversely, the dynamics of the droplets are essentially local and we use simple
analytical expressions for the droplet growth rate and the drift speed.
These expressions depend on the local supersaturation of the background in the
vicinity of the droplet.
Taken together, the dynamics of the emulsion is described by a reduced set of
dynamical degrees of freedom, compared to the typical Cahn-Hilliard description.
"""

__version__ = "0.1"

from .elements import *
from .state import State
from .actors import *
from .simulation import Simulation
from .trackers import *
