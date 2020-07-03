"""
Supplies the base class for actors

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Dict, Any, Type, Callable, Tuple  # @UnusedImport
import logging
from abc import abstractmethod, ABCMeta

from pde.tools.parameters import Parameterized
from pde.tools.cache import objects_equal

from ..elements import ElementBase


ElementsType = Tuple[ElementBase, ...]


class ActorBase(Parameterized, metaclass=ABCMeta):
    """ represents a single actor, which affects one or more elements """

    element_classes: Tuple[Type[ElementBase], ...] = (ElementBase,)
    """ tuple: defines the elements this actor handles and in what order they
    need to be supplied. The default assumes a single generic element. If an
    actor affects multiple elements, this values needs to be sepcified."""

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
    def num_elements(self) -> int:
        """ int: the number of elements this actor affects. This value is 
        determined from the `element_classes` attribute """
        return len(self.element_classes)

    @property
    def info(self) -> Dict[str, Any]:
        """ dict: information about the actor """
        return {"class": self.__class__.__name__, "parameters": self.parameters}

    def copy(self) -> "ActorBase":
        """ returns a copy the actor """
        return self.__class__(self.parameters.copy())

    def estimate_dt(self, element_states: ElementsType) -> float:
        """ estimate the maximal time step for simulating this actor 
        
        Args:
            element_states (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects

        Returns:
            float: the maximal time step
        """
        raise NotImplementedError

    def _check_cache(self, element_states: ElementsType) -> None:
        """ checks whether the simulation needs to run :meth:`_update_cache`.
        
        Subclasses can defined `_update_cache` to populate `self._cache` with
        pre-computed data, which is then available in later.
        
        Args:
            element_states (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects
        """
        if hasattr(self, "_update_cache"):
            # the class uses a cache internally
            state_attributes = tuple(el.attributes for el in element_states)
            if not objects_equal(self._cache.get("state_attributes"), state_attributes):
                # the cache is out-of-date
                self._update_cache(element_states)  # type: ignore
                self._cache["state_attributes"] = state_attributes

    def make_evolver_numba(self, element_states: ElementsType) -> Callable:
        """ return a function evolve the state from time `t` to `t + dt`
        
        Args:
            *element_states (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects

        Returns:
            callable: A function with signature
                (state_data: :class:`numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, element_states: ElementsType, t: float, dt: float):
        """ evolve the state from time `t` to `t + dt`
        
        Args:
            element_states (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects
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
