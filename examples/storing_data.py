#!/usr/bin/env python3
"""
Storing data during simulation
==============================

Example of how to store data during a simulation.
"""

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

import sim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(3)]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("background", sim.DiffusionActor())
simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

# run simulation and store data periodically
tracker = sim.TrajectoryTracker(
    "trajectory.zip", interrupts=2, mode="truncate", info=simulation.info
)
simulation.run(t_range=10, tracker=tracker)

# retrieve data and plot last state
stored_data = sim.Trajectory("trajectory.zip")
print(stored_data.info)  # recover the auxillary information
stored_data[-1].plot()  # plot the last time point of the stored data
