#!/usr/bin/env python3
"""
Droplets with emitters
======================

Demonstrates how the dynamics of droplets can be affected by randomly positioned points
in the background fluid that emit extra material.
"""

from droplets import SphericalDroplet
from pde import UnitGrid

import emulsim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = emulsim.ScalarFieldElement(parameters={"grid": grid})
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = emulsim.SphericalDropletsElement.from_droplets(droplet_data)
state = emulsim.State({"background": background, "droplets": droplets})

# set up simulation
simulation = emulsim.Simulation(state)
simulation.add_actor(
    "background", emulsim.DiffusionActor(parameters={"diffusivity": 0.1})
)
positions = [grid.get_random_point() for _ in range(5)]
simulation.add_actor(
    "background", emulsim.EmittersActor({"positions": positions, "strengths": 10})
)
simulation.add_actor(("droplets", "background"), emulsim.SphericalDropletActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
