'''
Provides a simple actor that emit mass into a field

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Callable, Dict, Any, Type, Union  # @UnusedImport

import numpy as np

from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from .base import AutonomousActorBase
from ...elements.fields import FieldElementBase



class EmittersActor(AutonomousActorBase):
    """ represents agents that emit mass into the background field """


    parameters_default = [
        Parameter('positions', np.array(tuple()), np.array,
                  "The positions of the emitters"),
        Parameter('strengths', np.array([1]), np.array,
                  "The strengths of the emitters"),
    ]
    

    state_class = FieldElementBase
    
    
    def __len__(self):
        return len(self.parameters['positions'])
    
    
    def estimate_dt(self, element: FieldElementBase) -> float:  # type: ignore
        """ estimate the maximal time step for simulating this agent type 
        
        Args:
            agents_state (:class:`EmitterAgentsState`):
                The state corresponding to this agent type (not used)
            background_state \
                   (:class:`~agent_based.backgrounds.base.FieldStateBase`):
                The state corresponding to the background

        Returns:
            float: the maximal time step
        """
        return float('inf')


    def make_evolver_numba(self, element: FieldElementBase) -> Callable:  # type: ignore
        """ return a function evolve the agents state from time `t` to `t + dt`
        
        Args:
            agents_state (:class:`EmitterAgentsState`):
                The state of all the droplets        
            background_state (:class:`FieldStateBase`):
                The state of the background

        Returns:
            callable: A function with signature
                (agents_data: :class:`numpy.ndarray`, t: float, dt: float,
                background_data), evolving `agents_data`
        """
        add_amount = element.make_add_amount_compiled()
        
        positions = np.asarray(self.parameters['positions'])
        strengths = np.broadcast_to(self.parameters['strengths'], (len(positions), ))
        
        @jit
        def evolver(state_data: np.ndarray, t: float, dt: float):
            """ evolve all agents explicitly """
            for position, strength in zip(positions, strengths):
                add_amount(state_data, position, dt * strength)

        return evolver  # type: ignore


    def evolve(self, element: FieldElementBase, t: float, dt: float) -> None:
        """ evolve the agents state from time `t` to `t + dt`
        
        Args:
            agents_state (:class:`EmitterAgentsState`):
                The state of all the droplets        
            background_state (:class:`FieldStateBase`):
                The state of the background
            t (float):
                The current time point
            dt (float):
                The time step
        """
        positions = self.parameters['positions']
        strengths = np.broadcast_to(self.parameters['strengths'], (len(positions), ))
        for position, strength in zip(positions, strengths):
            element.add_amount(position, dt * strength)
