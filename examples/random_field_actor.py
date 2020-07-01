#!/usr/bin/env python3

import numpy as np
import numba as nb

from pde import UnitGrid
import sim
from sim.actors.autonomous.base import AutonomousActorBase


class RandomFieldActor(AutonomousActorBase):
    """ actor that sets a new random field each time step """


    def evolve(self, element, t, dt):
        # mandatory python implementation of the background evolution
        element.data = np.random.uniform(0, 0.1, element.data.shape)
        

    def make_evolver_numba(self, element):
        """ implementing the compiled version is optional """
        # this function is optional and can be used to speed up calculations
        @nb.jit
        def evolver(element_state, t: float, dt: float):
            """ evolve the diffusion equation explicitly """
            for i in range(element_state.size):
                element_state.flat[i] = np.random.uniform(0, 0.1)
        return evolver  # type: ignore
    

# setup state
element = sim.ScalarFieldElement(parameters={'grid': UnitGrid([32, 32])})
state = sim.State({'field': element})

# setup simulation
simulation = sim.Simulation(state)
simulation.add_actor('field', RandomFieldActor())

# run simulation
result = simulation.run(t_range=10)

result.plot()
