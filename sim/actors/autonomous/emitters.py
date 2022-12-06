"""
Provides a simple actor that emit mass into a field a predefined positions

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Callable, Dict, Tuple, Type, Union  # @UnusedImport

import numpy as np

from modelrunner.parameters import Parameter
from pde.tools.numba import jit

from ...elements.fields import FieldElementBase
from ..base import ActorBase, ElementsType


class EmittersActor(ActorBase):
    """represents actor that emit mass into a field at defined positions"""

    parameters_default = [
        Parameter(
            "positions",
            np.array(tuple()),
            np.array,
            "The positions of all the emitters. This needs to be an array of "
            "positions. The dimension of each position needs to be compatible with "
            "the dimension of the field.",
        ),
        Parameter(
            "strengths",
            np.array([1]),
            np.array,
            "The strengths of the emitters, i.e., the mass per unit time that is "
            "emitted. This can be an array, setting different strengths for each "
            "emitter, or a single number, setting the same strength for all emitters.",
        ),
    ]

    element_classes = (FieldElementBase,)

    def __len__(self):
        """int: return the number of dimensions"""
        return len(self.parameters["positions"])

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            element (:class:`~sim.elements.fields.FieldElementBase`):
                The field element that is effected by the emitters

        Returns:
            float: the maximal time step
        """
        return float("inf")

    def make_evolver_numba(
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
        """return a function evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The field element that is effected by the emitters

        Returns:
            callable: A function with signature
                (field_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `field_data`
        """
        (element,) = elements  # extract single element
        add_amount = element.make_add_amount_compiled()  # type: ignore

        positions = np.asarray(self.parameters["positions"])
        strengths = np.broadcast_to(self.parameters["strengths"], (len(positions),))

        @jit
        def evolver(state_data: Tuple[np.ndarray], t: float, dt: float):
            """evolve all emitters explicitly"""
            for position, strength in zip(positions, strengths):
                add_amount(state_data[0], position, dt * strength)

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The field element that is effected by the emitters
            t (float):
                The current time point
            dt (float):
                The time step
        """
        (element,) = elements  # extract single element
        positions = self.parameters["positions"]
        strengths = np.broadcast_to(self.parameters["strengths"], (len(positions),))
        for position, strength in zip(positions, strengths):
            element.add_amount(position, dt * strength)  # type: ignore
