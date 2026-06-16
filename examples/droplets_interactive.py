#!/usr/bin/env python3
"""
Show droplet dynamics interactively
===================================

Example of droplet coarsening visualized interactively
"""

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

import emulsim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = emulsim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 0.5) for _ in range(10)]
droplets = emulsim.SphericalDropletsElement.from_droplets(droplet_data)
state = emulsim.State({"background": background, "droplets": droplets})

# set up simulation
simulation = emulsim.Simulation(state)
simulation.add_actor("background", emulsim.DiffusionActor())
simulation.add_actor(("droplets", "background"), emulsim.SphericalDropletActor())

# run simulation with an interactive visualization
simulation.run(t_range=100, tracker=["progress", "interactive"])
