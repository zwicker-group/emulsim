"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pde.backends.numba.utils import jit
from pde.grids.cartesian import CartesianGrid

from ... import Parameter
from ...elements import ArrowsElement, PointsElement, SphericalDropletsElement
from ..base import ActorBase, ElementsType


class BoxActor(ActorBase):
    """Actor containing particles in a box."""

    parameters_default = [
        Parameter("bounds", [], np.array, "The bounds of the box"),
        Parameter("periodic", False, np.array, "The bounds of the box"),
        Parameter(
            "point_like",
            True,
            bool,
            "When False, the radius of the object is used in the distance calculation",
        ),
    ]

    element_classes = ((PointsElement, ArrowsElement, SphericalDropletsElement),)

    def __init__(self, parameters: dict[str, Any] | None = None):
        """
        Args:
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~BoxActor.show_parameters` for details.
        """
        super().__init__(parameters=parameters)

        # convert periodicity information into useful format
        periodic = self.parameters["periodic"]
        if isinstance(periodic, np.ndarray) and periodic.size == 1:
            periodic = periodic.item()

        self._grid = CartesianGrid(self.parameters["bounds"], 1, periodic)

    @classmethod
    def from_grid(cls, grid: CartesianGrid):
        """Create BoxActor from a Cartesian grid.

        Args:
            grid (:class:`pde.grids.cartesian.CartesianGrid`):
                The Cartesian grid that defines the box
        """
        return cls({"bounds": grid.axes_bounds, "periodic": grid.periodic})

    def estimate_dt(self, elements: ElementsType) -> float:
        """Estimate the maximal time step for simulating this actor.

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
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
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The element that is affected by this actor

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        (points_element,) = elements  # extract single element

        num_points = len(points_element.data)
        num_axes = self._grid.num_axes
        periodic = np.array(self._grid.periodic)  # using a tuple led to a numba error
        bounds = np.array(self._grid.axes_bounds)
        midpoint = self._grid.cuboid.centroid
        xmin = bounds[:, 0]
        xmax = bounds[:, 1]
        size = bounds[:, 1] - bounds[:, 0]

        # figure out which axes need to be considered for flipping direction
        if "direction" in points_element.data.dtype.fields:
            flip_ax: np.ndarray = np.flatnonzero(np.logical_not(self._grid.periodic))
        else:
            flip_ax = np.empty((0,))
        test_for_flipping = flip_ax.size > 0

        point_like = self.parameters["point_like"]

        @jit
        def evolver(state_data: tuple[np.ndarray], t: float, dt: float) -> None:
            """Evolve all points explicitly."""
            points = state_data[0]  # data of the points
            for i in range(num_points):
                pos = points[i].position

                if point_like:
                    radius = 0
                else:
                    radius = points[i].radius

                # flip direction if out of bound
                if test_for_flipping:
                    for ax in flip_ax:
                        dist_norm = (pos[ax] - midpoint[ax]) / (size[ax] - 2 * radius)
                        if (dist_norm - 0.5) % 2 - 1 < 0:
                            points[i].direction[ax] *= -1
                # TODO: this function's performance could be improved by calculating
                # the distance only once

                # move the points to inside the box
                for ax in range(num_axes):
                    if periodic[ax]:
                        pos[ax] = (pos[ax] - xmin[ax]) % size[ax] + xmin[ax]
                    else:
                        dist_left = pos[ax] - (xmax[ax] - radius)
                        size_red = size[ax] - 2 * radius
                        arg = (dist_left) % (2 * size_red) - size_red
                        pos[ax] = xmin[ax] + radius + abs(arg)

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """Evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The element that is affected by this actor
            t (float):
                The current time point
            dt (float):
                The time step
        """
        if not self.parameters["point_like"]:
            raise NotImplementedError("numpy backend can only deal with point-objects")

        (points,) = elements  # extract single element

        if "direction" in points.data.dtype.fields:
            # flip direction if out of bound
            midpoint = self._grid.cuboid.centroid
            size = np.array(self._grid.cuboid.size)
            if not self.parameters["point_like"]:
                size = size[:, np.newaxis] - 2 * points.radius[np.newaxis, :]  # type: ignore

            for ax in range(points.dim):  # type: ignore
                if self._grid.periodic[ax]:
                    continue  # do nothing for periodic axes
                dist_norm = (points.positions[..., ax] - midpoint[ax]) / size[ax]  # type: ignore
                factor = np.sign((dist_norm - 0.5) % 2 - 1)
                factor[factor == 0] = 1  # don't flip corner cases
                points.directions[..., ax] *= factor  # type: ignore

        # move the points to inside the box
        # TODO: support extended objects, where `point_like` is False
        points.positions[...] = self._grid.normalize_point(  # type: ignore
            points.positions,  # type: ignore
            reflect=True,
        )
