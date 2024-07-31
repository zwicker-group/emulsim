#!/usr/bin/env python3
"""
Random field actor
==================

Demonstrates a custom actor class that sets a field to random values 
"""

import numba as nb
import numpy as np

from pde import UnitGrid

import sim


class RandomFieldActor(sim.ActorBase):
    """Actor that sets a new random field each time step."""

    def evolve(self, elements, t, dt):
        # mandatory python implementation of the background evolution
        (field,) = elements
        field.data = np.random.uniform(0, 0.1, field.data.shape)

    def make_evolver_numba(self, elements):
        """Implementing the compiled version is optional."""

        # this function is optional and can be used to speed up calculations
        @nb.jit
        def evolver(elements_state, t: float, dt: float):
            """Evolve the diffusion equation explicitly."""
            (field_state,) = elements_state
            for i in range(field_state.size):
                field_state.flat[i] = np.random.uniform(0, 0.1)

        return evolver  # type: ignore


# set up state
element = sim.ScalarFieldElement(parameters={"grid": UnitGrid([32, 32])})
state = sim.State({"field": element})

# set up simulation
simulation = sim.Simulation(state)
simulation.add_actor("field", RandomFieldActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
