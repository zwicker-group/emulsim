#!/usr/bin/env python3

import numpy as np

import sim

# setup state
particle_data = np.random.uniform(0, 100, size=(10, 2))
particles = sim.ArrowsElement.from_position_random_direction(particle_data)
state = sim.State({"particles": particles})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor(
    "particles", sim.ActiveParticleActor({"rotational_diffusion": 1.0})
)

# run simulation
result = simulation.run(t_range=10)

result.plot()
