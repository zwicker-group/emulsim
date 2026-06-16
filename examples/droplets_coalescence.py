#!/usr/bin/env python3
"""
Brownian droplet coarsening
===========================

Simple examples of droplets subjected to Brownian motion and coalescence.
"""

import numpy as np

import pde
from droplets import SphericalDroplet

import emulsim

# set up state
droplets = [SphericalDroplet(50 * np.random.randn(2), 5) for i in range(100)]
state = emulsim.State(
    {"droplets": emulsim.SphericalDropletsElement.from_droplets(droplets)}
)

# set up simulation
simulation = emulsim.Simulation(state)
simulation.add_actor(
    "droplets", emulsim.BrownianMotionActor({"diffusivity": "1/radius"})
)
simulation.add_actor("droplets", emulsim.CoalescenceDropletActor())

# run simulation
result = simulation.run(t_range=1e3, dt=1, tracker=pde.PlotTracker(10, show=True))

result.plot()
