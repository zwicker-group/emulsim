#!/usr/bin/env python3

import numpy as np

import sim


class BrownianParticlesActor(sim.AutonomousActorBase):
    
    diffusivity = 1
    
    def evolve(self, particles, t, dt):
        """ evolve the particles in time """
        scale = np.sqrt(dt) * self.diffusivity
        particles.data += scale * np.random.normal(size=particles.data.shape)
            


# setup state
particle_data = np.random.uniform(0, 100, size=(10, 2))
particles = sim.PointsElement(particle_data)
state = sim.State({'particles': particles})


# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor('particles', BrownianParticlesActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
