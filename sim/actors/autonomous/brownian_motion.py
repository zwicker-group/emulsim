"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple

import numba as nb
import numpy as np

from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import PointsElement, SphericalDropletsElement
from ..base import ActorBase, ElementsType


class BrownianMotionPointActor(ActorBase):
    """ represents actor that moves points according to Brownian motion """

    parameters_default = [
        Parameter(
            "diffusivity",
            1,
            float,
            "The diffusivity of the particles, which determines the strength of the "
            "Brownian motion.",
        ),
    ]

    element_classes = (PointsElement,)

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The points element that is affected by the Brownian motion

        Returns:
            float: the maximal time step
        """
        return float("inf")

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray], float, float], None]:
        """return a function evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The points element that is affected by the Brownian motion

        Returns:
            callable: A function with signature
                (field_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `field_data`
        """
        if isinstance(elements[0], SphericalDropletsElement):
            self._logger.warning(
                f"Using {self.__class__.__name__} to act on SphericalDropletsElement "
                "instead of BrownianMotionDropletActor!"
            )

        diffusivity = self.parameters["diffusivity"]

        @jit
        def evolver(state_data: Tuple[np.ndarray], t: float, dt: float) -> None:
            """ evolve all points explicitly """
            scale = np.sqrt(dt * diffusivity)
            for i in nb.prange(state_data[0].size):
                state_data[0].flat[i] += scale * np.random.randn()  # type: ignore

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The points element that is affected by the Brownian motion
            t (float):
                The current time point
            dt (float):
                The time step
        """
        (points,) = elements  # extract single element
        scale = np.sqrt(dt * self.parameters["diffusivity"])
        points.data[...] += scale * np.random.randn(*points.data.shape)


class BrownianMotionDropletActor(ActorBase):
    """ represents actor that moves droplets according to Brownian motion """

    parameters_default = [
        Parameter(
            "diffusivity",
            "1",
            str,
            "Expression that determines the strength of the Brownian motion of "
            "droplets. The expression may depend on the droplet radius and time.",
        ),
    ]

    element_classes = (SphericalDropletsElement,)

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that is effected by the Brownian motion

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
        self._cache["diffusivity"] = ScalarExpression(
            self.parameters["diffusivity"], [["radius", "R"], ["time", "t"]]
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
                (field_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `field_data`
        """
        if isinstance(elements[0], PointsElement):
            self._logger.warning(
                f"Using {self.__class__.__name__} to act on PointsElement instead of "
                "BrownianMotionPointActor!"
            )

        self._check_cache(elements)
        diffusivity = self._cache["diffusivity"].get_compiled()
        dim = elements[0].dim

        @jit
        def evolver(state_data: Tuple[np.ndarray], t: float, dt: float):
            """ evolve all points explicitly """
            (droplets_data,) = state_data
            for droplet_data in droplets_data:
                if droplet_data.radius > 0:
                    scale = np.sqrt(dt * diffusivity(droplet_data.radius, t))
                    for i in range(dim):
                        droplet_data.position[i] += scale * np.random.randn()

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that is effected by the Brownian motion
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(elements)
        (droplets,) = elements  # extract single element
        diffusivity = self._cache["diffusivity"]
        dim = droplets.dim

        for droplet in droplets.droplets:  # type: ignore
            if droplet.radius > 0:
                scale = np.sqrt(dt * diffusivity(droplet.radius, t))
                droplet.position += scale * np.random.randn(dim)
