"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable

import numba as nb
import numpy as np

from pde.backends.numba.utils import jit

from ... import Parameter
from ...elements import ArrowsElement
from ..base import ActorBase, ElementsType


class ActiveParticleActor(ActorBase):
    """Actor moving arrows according to their direction."""

    parameters_default = [
        Parameter(
            "rotational_diffusion", 0.0, float, "The rotational diffusion strength"
        )
    ]

    element_classes = (ArrowsElement,)

    def estimate_dt(self, elements: ElementsType) -> float:
        """Estimate the maximal time step for simulating this actor.

        Args:
            elements (tuple of :class:`~emulsim.elements.points.ArrowsElement`):
                The element that is affected by the directed motion

        Returns:
            float: the maximal time step
        """
        return float("inf")

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[tuple[np.ndarray], float, float], None]:
        """Return a function evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~emulsim.elements.points.ArrowsElement`):
                The element that is affected by the directed motion

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        dim = int(elements[0].dim)  # type: ignore

        rot_diff = float(self.parameters["rotational_diffusion"])
        if rot_diff > 0 and dim > 2:
            raise NotImplementedError

        @jit
        def evolver(state_data: tuple[np.ndarray], t: float, dt: float) -> None:
            """Evolve all points explicitly."""
            points = state_data[0]

            for i in nb.prange(len(state_data[0])):
                # update the position
                for j in range(dim):
                    points[i].position[j] += dt * points[i].direction[j]

                # apply rotational diffusion if requested
                if rot_diff > 0:
                    if dim == 1:
                        # interpret rot_diff as rate of flipping
                        if np.random.rand() < dt * rot_diff:
                            points[i].direction[:] *= -1.0

                    elif dim == 2:
                        # rotate by angle chosen from normal distribution
                        φ = np.random.normal(0, dt * rot_diff)
                        cosφ, sinφ = np.cos(φ), np.sin(φ)
                        dx, dy = points[i].direction
                        points[i].direction[0] = cosφ * dx - sinφ * dy
                        points[i].direction[1] = sinφ * dx + cosφ * dy

                    else:
                        # higher dimensions are not currently supported
                        raise NotImplementedError

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """Evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~emulsim.elements.points.ArrowsElement`):
                The element that is affected by the directed motion
            t (float):
                The current time point
            dt (float):
                The time step
        """
        (points,) = elements  # extract single element

        # update the position
        points.positions += dt * points.directions  # type: ignore

        # apply rotational diffusion if requested
        rot_diff = self.parameters["rotational_diffusion"]
        if rot_diff > 0:
            if points.dim == 1:
                # interpret rot_diff as rate of flipping
                flip = np.random.rand(len(points.data)) < dt * rot_diff
                points.directions[flip] *= -1  # type: ignore

            elif points.dim == 2:
                # rotate by angle chosen from normal distribution
                φ = np.random.normal(0, dt * rot_diff, size=len(points.data))
                rot_mat = np.array([[np.cos(φ), -np.sin(φ)], [np.sin(φ), np.cos(φ)]])
                new_direction = np.einsum("pi,ijp->pj", points.directions, rot_mat)  # type: ignore
                points.directions[:] = new_direction  # type: ignore

            else:
                raise NotImplementedError(
                    "Rotational diffusion is not implemented for points with "
                    f"{points.dim} dimensions."
                )
