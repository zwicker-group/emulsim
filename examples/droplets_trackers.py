#!/usr/bin/env python3
r"""
Trackers in active droplet simulations
======================================

This example shows how to analyze droplet trajectories. Note that the simulation
requires the optional `numba-scipy` package!
"""

import numba_scipy  # raises an error if the module is not available
import numpy as np

from droplets import SphericalDroplet
from pde import ScalarField, UnitGrid

import sim

# set up state
grid = UnitGrid([32, 32], periodic=True)
background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.005))
droplet_data = [
    SphericalDroplet(position=grid.get_random_point(), radius=np.random.uniform(0.1, 4))
    for _ in range(10)
]
droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
state = sim.State({"background": background, "droplets": droplets})

# set up simulation
reaction_flux = "0.001 - 0.01 * c"
simulation = sim.Simulation(state)
simulation.add_actor(
    "background", sim.ReactionDiffusionActor({"reaction_flux": reaction_flux})
)
droplet_actor = sim.SphericalDropletActor(
    {"mean_reaction_inside": -0.01, "reaction_outside": reaction_flux}
)
simulation.add_actor(("droplets", "background"), droplet_actor)

# run simulation
tracker = sim.DropletElementTracker("droplets", interrupts=10)
simulation.run(t_range=1e3, tracker=tracker)

# show droplet tracks
tracker.droplet_tracks.plot()
