#!/usr/bin/env python3
"""
Custom Brownian motion class 
============================

Demonstrates the custom implementation of Brownian motion.
"""

import numpy as np

import sim


class BrownianParticlesActor(sim.ActorBase):
    diffusivity = 1

    def evolve(self, elements, t, dt):
        """Evolve the particles in time."""
        (particles,) = elements
        scale = np.sqrt(dt) * self.diffusivity
        size = particles.positions.shape
        particles.positions[...] += scale * np.random.normal(size=size)


# set up state
particle_data = np.random.uniform(0, 100, size=(10, 2))
particles = sim.PointsElement(particle_data)
state = sim.State({"particles": particles})

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("particles", BrownianParticlesActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
