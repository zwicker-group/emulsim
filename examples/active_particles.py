#!/usr/bin/env python3
"""
Active particles
================

Simple implementation of non-interacting active particles.
"""

import numpy as np

import sim

# set up state
particle_data = np.random.uniform(0, 100, size=(10, 2))
particles = sim.ArrowsElement.from_position_random_direction(particle_data)
state = sim.State({"particles": particles})

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor(
    "particles", sim.ActiveParticleActor({"rotational_diffusion": 1.0})
)
simulation.add_actor("particles", sim.BoxActor({"bounds": [[0, 100], [0, 100]]}))

# run simulation
result = simulation.run(t_range=10)

result.plot()
