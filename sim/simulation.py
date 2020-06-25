'''
Provides a class representing the full simulation

.. autosummary::
   :nosignatures:

   ~Simulation
   ~SimulationSolver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import logging
from typing import Callable, Any, Dict, Union, Tuple

import numpy as np

from pde.solvers.base import SolverBase
from pde.solvers.controller import (Controller, TRangeType,
                                    TrackerCollectionDataType)
from pde.tools.numba import jit

from .actors.base import ActorBase
from .state import State



class Simulation():
    """ Class defining the agent-based simulation """
    
    def __init__(self, state, actors=None):
        """
        Args:
            background \
                   (:class:`~agent_based.backgrounds.base.BackgroundStateBase`):
                The instance describing the background
            agents (:class:`~agent_based.agents.base.AgentsBase`):
                The instance describing all the agents. If omitted, no agents
                are added.
        """
        self.state = state
        self._logger = logging.getLogger(self.__class__.__name__)
        self.actors = []
        if actors is not None:
            for element_names, actor in actors:
                self.add_actor(element_names, actor)
    
    
    def __repr__(self):
        """ return instance as string """
        actors_str = ', '.join(repr(actor) for actor in self.actors)
        return (f'{self.__class__.__name__}({self.state!r}, actors=[{actors_str}])')

        
    def __str__(self):
        """ return instance as string """
        actors_str = ', '.join(str(actor) for actor in self.actors)
        return (f'{self.__class__.__name__}({self.state!s}, actors=[{actors_str}])')

        
    @property
    def info(self) -> Dict[str, Any]:
        """ dict: information about the state """
        actor_infos = []
        for element_names, actor in self.actors:
            info = actor.info
            info['element_names'] = element_names
            actor_infos.append(info)
        return {'state': self.state.attributes,
                'actors': actor_infos}
        
        
    def add_actor(self, elements: Union[str, Tuple[str]], actor: ActorBase):
        """ adds a new actor to the simulation """
        if isinstance(elements, str):
            elements = (elements,)
            
        assert len(elements) == actor.num_elements
        self.actors.append((elements, actor))
        
        
    def estimate_dt(self, state: State) -> float:
        """ get the optimal time step for the simulation
                
        Returns:
            float: the time step
        """
        dts: List[float] = [np.inf]
        for elements, actor in self.actors:
            try:
                dt = actor.estimate_dt(*state[elements])
            except NotImplementedError:
                self._logger.info('Could not determine time step for actor '
                                  f'"{actor}"')
            else:
                dts.append(dt)
                
        return min(dts)
        

    def make_evolver_numba(self, state: State) -> Callable:
        """ return a function evolving the state from time `t` to `t + dt`
        
        Args:
            state (:class:`SimulationState`):
                The state of the simulation
                
        Returns:
            callable: A function with signature (state_data, t: float,
            dt: float), which evolves the state in time
        """
        actors = []
        for elements, actor in self.actors:
            actor_data = {'actor': actor,
                          'evolver': actor.make_evolver_numba(*state[elements]),
                          'element_indices': tuple(state.get_index(name)
                                                   for name in elements)}
            actors.append(actor_data)
        
        @jit
        def innermost(state_data, t, dt):
            """ no-op function serving as innermost nested function """
            pass
                
        def chain(actor_id, inner) -> Callable:
            """ recursive helper function for evolving all agents """
            # run through all evolvers
            evolver = actors[actor_id]['evolver']
            element_indices = actors[actor_id]['element_indices']
            num_elements = len(element_indices)

            if num_elements == 1:
                i = element_indices[0]
                @jit
                def wrap(state_data, t: float, dt: float):
                    inner(state_data, t, dt)
                    evolver(state_data[i], t, dt)
            
            elif num_elements == 2:
                i, j = element_indices
                @jit
                def wrap(state_data, t: float, dt: float):
                    inner(state_data, t, dt)
                    evolver(state_data[i], state_data[j], t, dt)
            
            if actor_id < len(actors) - 1:
                # there are more items in the chain
                return chain(actor_id + 1, inner=wrap)
            else:
                # this is the outermost function
                return wrap  # type: ignore
        
        # compile the recursive chain
        return chain(0, innermost)
    
    
    def evolve(self, state: State, t: float, dt: float):
        """ evolve the state from time `t` to `t + dt`
        
        Args:
            state (:class:`SimulationState`):
                The state of the simulation
            t (float):
                The current time point
            dt (float):
                The time step
        """        
        for elements, actor in self.actors:
            actor.evolve(*state[elements], t, dt)
        

    def run(self,
            t_range: TRangeType,
            dt: float = None, 
            tracker: TrackerCollectionDataType = ['progress'],
            backend: str = 'auto') -> State:
        """ run an agent-based simulation
        
        Args:
            state (:class:`SimulationState`):
                The initial state
            t_range (float or tuple of floats):
                Sets the time range for which the simulation is run. If only a
                single value `t_end` is given, the time range is assumed to be 
                `[0, t_end]`.
            dt (float):
                Time step of the explicit stepping. If `None`, the time step
                will be chosen automatically using the method
                :func:`~agent_based.state.AgentSimulation.estimate_dt`.
            tracker:
                Defines trackers that process the state of the simulation at
                fixed time intervals. Multiple trackers can be specified as a
                list. The default value simply displays a progress bar. To
                disable trackers, set the value to `None`.        
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
            
        Returns:
            :class:`SimulationState`:
                The state of the simulation at the last time point 
        """
        solver = SimulationSolver(self, backend=backend)
        controller = Controller(solver, t_range=t_range, tracker=tracker)
        return controller.run(self.state, dt)  # type: ignore



class SimulationSolver(SolverBase):
    """ Solver for agent-based simulation of emulsions """


    def __init__(self, simulation: Simulation, backend: str = 'auto'):
        """ initialize the explicit solver for the agent-based simulation
        
        Args:
            simulation (:class:`~agent_based.simulation.AgentSimulation`):
                The simulation that will be run. This defines the behavior of
                the background and the agents.
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
        """        
        self.info: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self.simulation = simulation
        self.backend = backend


    def _make_stepper_numpy(self, dt: float) -> Callable:
        """ return function evolving state from time `t_start` to `t_end`
        
        Args:
            dt (float): The time step
            
        Returns:
            callable: Function with signature (state: SimulationState,
            t_start: float, t_end: float), which advances `state` in time.
        """

        def stepper(state: State, t_start: float, t_end: float) \
                -> float:
            """ function that advances the state from t_start to t_end """
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                self.simulation.evolve(state, t, dt)
                
            self.info['steps'] += steps

            return t + dt
        
        return stepper


    def _make_stepper_numba(self, state: State, dt: float) \
            -> Callable:
        """ return function evolving state from time `t_start` to `t_end`
        
        This function used compiled evolvers for the background and the agents.
        
        Args:
            state (:class:`~agent_based.simulation.SimulationState`):
                An example of the simulation state
            dt (float):
                The time step
                
        Returns:
            callable: Function with signature (state: SimulationState,
            t_start: float, t_end: float), which advances `state` in time.
        """
        simulation_evolver = self.simulation.make_evolver_numba(state)

        def stepper(state: State, t_start: float, t_end: float) \
                -> float:
            """ function that advances the state from t_start to t_end """
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                simulation_evolver(state.data, t, dt)

            self.info['steps'] += steps

            return t + dt

        return stepper


    def make_stepper(self, state: State, dt: float = None) \
            -> Callable:
        r""" return a stepper function using an explicit scheme
        
        Note that if the `numba` backend is chosen, the state supplied to this
        function must be the identical state that is also used in the stepper.  
        
        Args:
            state (:class:`~agent_based.simulation.SimulationState`):
                An example of the simulation state, which is used to extract the
                grid and other information.
            dt (float):
                Time step of the explicit stepping. If `None`, the time step
                will be chosen automatically using the method
                :func:`~agent_based.state.AgentSimulation.estimate_dt`.
            \**kwargs: These are currently ignored

        Returns:
            Function that can be called to advance the `state` from time
            `t_start` to time `t_end`. The function call signature is
            `(state: AgentSimulation, t_start: float, t_end: float)`        
        """
#         self.simulation.agents._prepare_evolver(state.agents,state.background)
        
        if dt is None:
            dt = self.simulation.estimate_dt(state)
            if np.isinf(dt):
                # this can happen if there are no restrictions on the time step
                dt = 1e3

        # store information about the simulation
        self.info['dt'] = dt
        self.info['steps'] = 0

        if self.backend == 'auto':
            try:
                return self._make_stepper_numba(state, dt)
            except NotImplementedError:
                self._logger.warning('Numba backend is not implemented for all '
                                     'parts of the simulation.')
                return self._make_stepper_numpy(dt)
                
        elif self.backend == 'numba':
            return self._make_stepper_numba(state, dt)
        elif self.backend == 'numpy':
            return self._make_stepper_numpy(dt)
        else:
            raise ValueError(f'Unknown backend `{self.backend}`')

