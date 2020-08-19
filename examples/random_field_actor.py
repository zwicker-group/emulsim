#!/usr/bin/env python3

import numba as nb
import numpy as np

from pde import UnitGrid

import sim


class RandomFieldActor(sim.ActorBase):
    """ actor that sets a new random field each time step """

    def evolve(self, elements, t, dt):
        # mandatory python implementation of the background evolution
        (field,) = elements
        field.data = np.random.uniform(0, 0.1, field.data.shape)

    def make_evolver_numba(self, elements):
        """ implementing the compiled version is optional """
        # this function is optional and can be used to speed up calculations
        @nb.jit
        def evolver(elements_state, t: float, dt: float):
            """ evolve the diffusion equation explicitly """
            (field_state,) = elements_state
            for i in range(field_state.size):
                field_state.flat[i] = np.random.uniform(0, 0.1)

        return evolver  # type: ignore


# setup state
element = sim.ScalarFieldElement(parameters={"grid": UnitGrid([32, 32])})
state = sim.State({"field": element})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor("field", RandomFieldActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
