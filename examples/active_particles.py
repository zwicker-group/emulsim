#!/usr/bin/env python3
"""
Active particles
================

Simple implementation of non-interacting active particles.
"""

import numpy as np

import emulsim

# set up state
particle_data = np.random.uniform(0, 100, size=(10, 2))
particles = emulsim.ArrowsElement.from_position_random_direction(particle_data)
state = emulsim.State({"particles": particles})

# set up simulation
simulation = emulsim.Simulation(state)
simulation.add_actor(
    "particles", emulsim.ActiveParticleActor({"rotational_diffusion": 1.0})
)
simulation.add_actor("particles", emulsim.BoxActor({"bounds": [[0, 100], [0, 100]]}))

# run simulation
result = simulation.run(t_range=10)

result.plot()
