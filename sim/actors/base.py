"""
Supplies the base class for actors

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import itertools
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

import numpy as np

from pde.tools.cache import objects_equal
from pde.tools.parameters import Parameterized

from ..elements import ElementBase

ElementsType = Tuple[ElementBase, ...]
ElementsSpec = Union[Type[ElementBase], List[Type[ElementBase]]]
EvolverType = Callable[[Tuple[np.ndarray, ...], float, float], None]


class ActorBase(Parameterized, metaclass=ABCMeta):
    """represents a single actor, which affects one or more elements"""

    element_classes: Tuple[ElementsSpec, ...] = tuple()
    """ tuple: defines the elements this actor handles and in what order they need to be
    supplied. An empty list indicates that all elements and lists of elements are
    accepted. Setting this attribute allows internal consistency checks."""

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
        """int: the number of elements this actor affects. This value is
        determined from the `element_classes` attribute"""
        return len(self.element_classes)

    @classmethod
    def supports_elements(
        cls,
        *elements: Union[ElementBase, Type[ElementBase]],
        silent: bool = False,
    ) -> bool:
        """determines whether this actor supports the given elements

        Args:
            elements (:class:`~sim.elements.base.ElementBase`):
                Various elements or element classes.
            silent (bool):
                Determines whether the function returns silently or whether an exception
                is raised when an element is not supported.

        Returns:
            bool: Whether the current actor supports the elements in the given order
        """
        if len(elements) != len(cls.element_classes):
            if silent:
                return False
            else:
                raise ValueError(f"Expected {len(cls.element_classes)} elements")

        # check whether all elements have the expected type
        for given, expected in zip(elements, cls.element_classes):
            if isinstance(given, ElementBase):
                given_cls = given.__class__
            elif inspect.isclass(given):
                given_cls = given
            else:
                raise TypeError("Instance of subclass of `ElementBase` required")

            # check whether the actor declares the element as matching
            mismatch_cls = None
            if hasattr(expected, "__iter__"):
                # actor supports multiple classes for this element
                if not any(issubclass(given_cls, cls) for cls in expected):  # type: ignore
                    mismatch_cls = ", ".join(cls.__name__ for cls in expected)  # type: ignore
            else:
                # actor supports a single class for this element
                if not issubclass(given_cls, expected):  # type: ignore
                    mismatch_cls = expected.__name__  # type: ignore

            if mismatch_cls:
                # the actor did not accept the element
                # check whether the element declares the actor as compatible
                if cls in given_cls._compatible_actors:
                    mismatch_cls = None

            if mismatch_cls:
                if silent:
                    return False
                else:
                    raise TypeError(
                        f"Element is a `{given_cls.__name__}`, but actor type "
                        f"`{cls.__name__}` expects `{mismatch_cls}`."
                    )
        return True

    @property
    def info(self) -> Dict[str, Any]:
        """dict: information about the actor"""
        return {"class": self.__class__.__name__, "parameters": self.parameters}

    def copy(self) -> ActorBase:
        """returns a copy the actor"""
        return self.__class__(self.parameters.copy())

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects

        Returns:
            float: the maximal time step
        """
        raise NotImplementedError

    def _check_cache(self, elements: ElementsType, **kwargs) -> None:
        r"""checks whether the simulation needs to run :meth:`_update_cache`.

        Subclasses can defined `_update_cache` to populate `self._cache` with
        pre-computed data, which is then available in later.

        Args:
            elements (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects
            \**kwargs:
                Additional arguments will be forwarded to the update cache function
        """
        if hasattr(self, "_update_cache"):
            # the class uses a cache internally
            state_attributes = tuple(el.attributes for el in elements)
            cache_key = state_attributes + tuple(sorted(kwargs.items()))
            if not objects_equal(self._cache.get("cache_key"), cache_key):
                # the cache is out-of-date
                self._update_cache(elements, **kwargs)  # type: ignore
                self._cache["cache_key"] = cache_key

    def make_evolver_numba(self, elements: ElementsType) -> EvolverType:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            *elements (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, elements: ElementsType, t: float, dt: float):
        """evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.base.ElementBase`):
                The elements that this actor affects
            t (float):
                The current time point
            dt (float):
                The time step

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                which evolves the state
        """
        pass


def find_actors(
    *elements: Union[ElementBase, Type[ElementBase]], unordered: bool = False
) -> List[Type[ActorBase]]:
    """finds actors compatible with the given elements

    Args:
        elements (:class:`~sim.elements.base.ElementBase`):
            Element classes or instances that shall serve as input for the actors
        unordered (bool):
            Determines whether also actors are returned that only accept a reordered
            arrangement of the elements.

    Returns:
        list of :class:`ActorBase`: A list of all compatible actor classes
    """
    # determine all tested permutations of the elements
    if unordered:
        elements_list = list(itertools.permutations(elements))
    else:
        elements_list = [elements]

    # check all actors
    result: Set[Type[ActorBase]] = set()
    for actor in ActorBase._subclasses.values():
        if issubclass(actor, ActorBase):
            for elements in elements_list:
                if actor.supports_elements(*elements, silent=True):
                    result.add(actor)
                    break

    # return a sorted list of actor classes
    return sorted(result, key=lambda cls: cls.__name__)
