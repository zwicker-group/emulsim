#!/usr/bin/env python3
"""
Biased droplet dynamics
=======================

Demonstrates how droplet dynamics can be biased using an external field, which is here
read from an image.
"""

from numba import jit

from droplets import SphericalDroplet
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.backends import backends

import sim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})


# define how the equilibrium concentration is calculated
class cEqOutField:
    """Class calculating equilibrium concentrations based on a field."""

    def __init__(self, image: ScalarField):
        self.image = image

    def __call__(self, position, radius, droplet_id):
        """Return equilibrium concentration for a position."""
        return 1e-2 * self.image.interpolate(position)

    def get_function(self, backend):
        """Return compiled function for calculating c_eq for a position."""
        interpolate = backends["numba"].make_interpolator(self.image)
        image_data = self.image.data

        @jit
        def c_eq(position, radius, droplet_id):
            return 1e-2 * interpolate(position, image_data)

        return c_eq


# set up random field used as the defining image
image_grid = CartesianGrid(grid.axes_bounds, 16, periodic=grid.periodic)
image = ScalarField.random_uniform(image_grid)

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("background", sim.DiffusionActor())
parameters = {"equilibrium_concentration": cEqOutField(image)}
simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor(parameters))

# run simulation
result = simulation.run(t_range=10)

result.plot()
