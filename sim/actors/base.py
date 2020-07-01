'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Dict, Any, Type, Callable  # @UnusedImport
import logging
from abc import abstractmethod, ABCMeta

from pde.tools.parameters import Parameterized
from pde.tools.cache import objects_equal



class ActorBase(Parameterized, metaclass=ABCMeta):
    """ represents the dynamics of many agents of the same type """

    num_elements: int  # the number of elements this actor affects


    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                Parameters defining the behavior of the actor. Call
                :meth:`~ActorBase.show_parameters` for details.
        """
        super().__init__(parameters)
        self._cache: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return objects_equal(self.info, other.info)
    
    
    @property
    def info(self) -> Dict[str, Any]:
        """ dict: information about the actor """
        return {'class': self.__class__.__name__,
                'parameters': self.parameters}


    def copy(self) -> "ActorBase":
        return self.__class__(self.parameters.copy())
    

    def estimate_dt(self, *element_states) -> float:
        """ estimate the maximal time step for simulating this agent type 
        
        Args:
            state (:class:`ElementBase`):
                The state corresponding to this actor

        Returns:
            float: the maximal time step
        """
        raise NotImplementedError
    
    
    def _check_cache(self, *element_states) -> None:
        """ checks whether the simulation needs to run :meth:`_update_cache`.
        
        Subclasses can defined `_update_cache` to populate `self._cache` with
        pre-computed data, which is then available in later.
        
        Args:
            state (:class:`ElementBase`):
                The state corresponding to this actor
        """
        if hasattr(self, '_update_cache'):
            # the class uses a cache internally
            state_attributes = tuple(el.attributes for el in element_states)
            if not objects_equal(self._cache.get('state_attributes'),
                                 state_attributes):
                # the cache is out-of-date
                self._update_cache(*element_states)  # type: ignore
                self._cache['state_attributes'] = state_attributes


    def make_evolver_numba(self, *element_states) -> Callable:
        """ return a function evolve the agents state from time `t` to `t + dt`
        
        Args:
            state (:class:`ElementBase`):
                The state corresponding to this actor

        Returns:
            callable: A function with signature
                (agents_data: :class:`numpy.ndarray`, t: float, dt: float,
                background_data), evolving `agents_data`
        """
        raise NotImplementedError


    @abstractmethod
    def evolve(self, element_states, t: float, dt: float):
        """ evolve the agents state from time `t` to `t + dt`
        
        Args:
            state (:class:`ElementBase`):
                The state corresponding to this actor
            t (float):
                The current time point
            dt (float):
                The time step

        Returns:
            callable: A function with signature
                (state_data: :class:`numpy.ndarray`, t: float, dt: float),
                which evolves the state
        """
        pass
    