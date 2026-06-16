#!/usr/bin/env python3
"""
Storing data during simulation
==============================

Example of how to store data during a simulation.
"""

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

import emulsim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = emulsim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = emulsim.SphericalDropletsElement.from_droplets(droplet_data)
state = emulsim.State({"background": background, "droplets": droplets})

# set up simulation
simulation = emulsim.Simulation(state)
simulation.add_actor("background", emulsim.DiffusionActor())
simulation.add_actor(("droplets", "background"), emulsim.SphericalDropletActor())

# run simulation and store data periodically
tracker = emulsim.TrajectoryTracker(
    "trajectory.zip", interrupts=2, mode="truncate", info=simulation.info
)
simulation.run(t_range=10, tracker=tracker)

# retrieve data and plot last state
stored_data = emulsim.Trajectory("trajectory.zip")
print(stored_data.info)  # recover the auxillary information
stored_data[-1].plot()  # plot the last time point of the stored data
