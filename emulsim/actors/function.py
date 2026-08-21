"""Provides a flexible actor based on a user-supplied python function.

.. autosummary::
   :nosignatures:

   ~FunctionActor
   ~NumbaFunctionActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import EllipsisType

from numba import TypingError

from pde.backends.numba.utils import jit

from ..elements.base import _ElementBase
from .base import ActorBase, ElementsSpec, ElementsType, EvolverType


class FunctionActor(ActorBase):
    """Actor that uses a python function to evolve elements."""

    element_classes: tuple[ElementsSpec, ...] | EllipsisType = Ellipsis

    def __init__(self, num_elements: int, func: Callable):
        """
        Args:
            num_elements (int):
                The number of elements this function expects
            func (callable):
                A python function that takes (elements, t, dt) as arguments and evolves
                the elements from time `t` to `t + dt`. `elements` is a tuple of the
                actual element classes. The function should not return anything. Better
                performance can be achieved when the function can be compiled by numba.
        """
        super().__init__()
        self.element_classes = (_ElementBase,) * num_elements

        # inspect and check that there are three arguments
        pos_args, unknown_args = 0, 0
        for p in inspect.signature(func).parameters.values():
            if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}:
                pos_args += 1
                if p.default is inspect._empty:
                    unknown_args += 1
        if pos_args < 3:
            raise ValueError("Function must accept at least 3 positional arguments")
        if unknown_args > 3:
            raise ValueError("Function must have at most 3 unspecified arguments")

        self.func = func

    def copy(self) -> FunctionActor:
        """Returns a copy the actor."""
        assert isinstance(self.num_elements, int)
        return self.__class__(self.num_elements, self.func)

    def evolve(self, elements: ElementsType, t: float, dt: float):
        """Evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~emulsim.elements.base._ElementBase`):
                The elements that this actor affects
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self.func(elements, t, dt)


class NumbaFunctionActor(ActorBase):
    """Actor that uses a compiled function to evolve the data of elements."""

    element_classes: tuple[ElementsSpec, ...] | EllipsisType = Ellipsis

    def __init__(self, num_elements: int, func: Callable):
        """
        Args:
            num_elements (int):
                The number of elements this function expects
            func (callable):
                A python function that takes (elements, t, dt) as arguments and evolves
                the elements from time `t` to `t + dt`. `elements` is a tuple of the
                data of elements. The function should not return anything. Better
                performance can be achieved when the function can be compiled by numba.
        """
        super().__init__()
        self.element_classes = (_ElementBase,) * num_elements

        # inspect and check that there are three arguments
        pos_args, unknown_args = 0, 0
        for p in inspect.signature(func).parameters.values():
            if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}:
                pos_args += 1
                if p.default is inspect._empty:
                    unknown_args += 1
        if pos_args < 3:
            raise ValueError("Function must accept at least 3 positional arguments")
        if unknown_args > 3:
            raise ValueError("Function must have at most 3 unspecified arguments")

        self.func = func

    def copy(self) -> NumbaFunctionActor:
        """Returns a copy the actor."""
        assert isinstance(self.num_elements, int)
        return self.__class__(self.num_elements, self.func)

    def make_evolver_numba(self, elements: ElementsType) -> EvolverType:
        """Return a function evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~emulsim.elements.base._ElementBase`):
                The elements that this actor affects

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        # run a quick test to see whether the function supports the correct arguments
        elements_data = tuple(el.copy(method="data").data for el in elements)
        self.func(elements_data, 10.0, 1e-3)  # test call with arbitrary t and dt

        # actually compile the function since it seemed to have passed the test
        try:
            return jit(self.func)
        except (RuntimeError, TypingError) as err:
            self._logger.warning("Could not compile user-supplied function")
            raise NotImplementedError from err

    def evolve(self, elements: ElementsType, t: float, dt: float):
        """Evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~emulsim.elements.base._ElementBase`):
                The elements that this actor affects
            t (float):
                The current time point
            dt (float):
                The time step
        """
        # extract data of elements to pass it to the user function
        element_data = [el.data for el in elements]
        self.func(element_data, t, dt)
