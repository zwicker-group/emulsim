#!/usr/bin/env python3

import numpy as np

from droplets import SphericalDroplet
import sim

# setup state
droplets = [SphericalDroplet(100 * np.random.randn(2), 5) for i in range(100)]
state = sim.State({'droplets': sim.SphericalDropletsElement.from_droplets(droplets)})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor('droplets', sim.BrownianMotionDropletActor())
simulation.add_actor('droplets', sim.CoalescenceDropletActor())

# run simulation
result = simulation.run(t_range=1e4, dt=1, tracker='plot')
