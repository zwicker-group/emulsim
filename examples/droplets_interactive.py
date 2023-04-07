#!/usr/bin/env python3
"""
Show droplet dynamics interactively
===================================

Example of droplet coarsening visualized interactively
"""

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

import sim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 0.5) for _ in range(10)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("background", sim.DiffusionActor())
simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

# run simulation with an interactive visualziation
simulation.run(t_range=100, tracker=["progress", "interactive"])
