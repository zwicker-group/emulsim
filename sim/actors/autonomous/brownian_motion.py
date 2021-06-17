"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple

import numpy as np

from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import ArrowsElement, PointsElement, SphericalDropletsElement
from ..base import ActorBase, ElementsType


class BrownianMotionActor(ActorBase):
    """represents actor that moves objects according to Brownian motion"""

    parameters_default = [
        Parameter(
            "diffusivity",
            "1",
            str,
            "Expression that determines the strength of the Brownian motion of "
            "droplets. The expression may depend on time and potentially on a radius "
            "if this is defined for the element on which the actor acts",
        ),
    ]

    element_classes = ((ArrowsElement, PointsElement, SphericalDropletsElement),)

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The element that is effected by the Brownian motion

        Returns:
            float: the maximal time step
        """
        return float("inf")

    def _update_cache(self, elements: ElementsType) -> None:
        """prepare the simulation doing pre-calculations

        Args:
            elements (tuple):
                The state of all the droplets and of the field
        """
        fields = elements[0].data.dtype.fields
        assert "position" in fields  # type: ignore
        self._cache["has_radius"] = "radius" in fields  # type: ignore
        if self._cache["has_radius"]:
            self._cache["diffusivity"] = ScalarExpression(
                self.parameters["diffusivity"], [["radius", "R"], ["time", "t"]]
            )
        else:
            self._cache["diffusivity"] = ScalarExpression(
                self.parameters["diffusivity"], [["time", "t"]]
            )

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray], float, float], None]:
        """return a function evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that is effected by the Brownian motion

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        self._check_cache(elements)
        diffusivity = self._cache["diffusivity"].get_compiled()
        dim = int(elements[0].dim)  # type: ignore

        if self._cache["has_radius"]:

            @jit
            def evolver(state_data: Tuple[np.ndarray], t: float, dt: float):
                """evolve all points explicitly"""
                (droplets_data,) = state_data
                for droplet_data in droplets_data:
                    if droplet_data.radius > 0:
                        scale = np.sqrt(dt * diffusivity(droplet_data.radius, t))
                        for i in range(dim):
                            droplet_data.position[i] += scale * np.random.randn()

        else:

            @jit
            def evolver(state_data: Tuple[np.ndarray], t: float, dt: float):
                """evolve all points explicitly"""
                (droplets_data,) = state_data
                scale = np.sqrt(dt * diffusivity(t))
                for droplet_data in droplets_data:
                    for i in range(dim):
                        droplet_data.position[i] += scale * np.random.randn()

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The element that is effected by the Brownian motion
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(elements)
        (objs,) = elements  # extract single element
        diffusivity = self._cache["diffusivity"]
        dim = objs.dim

        if self._cache["has_radius"]:
            for droplet in objs.droplets:  # type: ignore
                if droplet.radius > 0:
                    scale = np.sqrt(dt * diffusivity(droplet.radius, t))
                    droplet.position += scale * np.random.randn(dim)
        else:
            scale = np.sqrt(dt * diffusivity(t))
            objs.positions[...] += scale * np.random.randn(len(objs), objs.dim)  # type: ignore
