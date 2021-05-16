"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np

from pde.grids.cartesian import CartesianGrid, CartesianGridBase
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import ArrowsElement, PointsElement
from ..base import ActorBase, ElementsType


class BoxActor(ActorBase):
    """ represents actor that contains particles in a box """

    parameters_default = [
        Parameter("bounds", [], np.array, "The bounds of the box"),
        Parameter("periodic", False, np.array, "The bounds of the box"),
    ]

    element_classes = ((PointsElement, ArrowsElement),)

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~BoxActor.show_parameters` for details.
        """
        super().__init__(parameters=parameters)
        # self.bounds = np.atleast_2d(self.parameters["bounds"])
        # assert self.bounds.shape[1] == 2
        # self.dim = self.bounds.shape[0]
        # self.periodic = np.broadcast_to(self.parameters["periodic"], (self.dim,))
        self._grid = CartesianGrid(
            self.parameters["bounds"], 1, self.parameters["periodic"]
        )

    @classmethod
    def from_grid(cls, grid: CartesianGridBase):
        """create BoxActor from a Cartesian grid

        Args:
            grid (:class:`pde.grids.cartesian.CartesianGridBase`):
                The Cartesian grid that defines the box
        """
        return cls({"bounds": grid.axes_bounds, "periodic": grid.periodic})

    def estimate_dt(self, elements: ElementsType) -> float:
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The element that is affected by the directed motion

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
                The element that is affected by this actor

        Returns:
            callable: A function with signature
                (state_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `state_data`
        """
        (points_element,) = elements  # extract single element

        normalize_point = self._grid.make_normalize_point_compiled(reflect=True)
        num_points = len(points_element.data)
        dim = self._grid.dim
        midpoint = self._grid.cuboid.centroid
        size = self._grid.cuboid.size

        # figure out which axes need to be considered for flipping direction
        if isinstance(points_element, ArrowsElement):
            flip_ax = np.flatnonzero(np.logical_not(self._grid.periodic))
        else:
            flip_ax = np.empty((0,))
        test_for_flipping = flip_ax.size > 0

        @jit
        def evolver(state_data: Tuple[np.ndarray], t: float, dt: float) -> None:
            """ evolve all points explicitly """
            points = state_data[0]  # coordinates of the points
            for i in range(num_points):
                # TODO: this function's performance could be improved by calculating
                # the distance only once

                # flip direction if out of bound
                if test_for_flipping:
                    for ax in flip_ax:
                        dist_norm = (points[i, ax] - midpoint[ax]) / size[ax]
                        if (dist_norm - 0.5) % 2 - 1 < 0:
                            points[i, dim + ax] *= -1

                # move the points to inside the box
                normalize_point(points[i, :dim])

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.points.PointsElement`):
                The element that is affected by this actor
            t (float):
                The current time point
            dt (float):
                The time step
        """
        (points,) = elements  # extract single element
        pos = points.data[..., : points.dim]

        if isinstance(points, ArrowsElement):
            # flip direction if out of bound
            midpoint = self._grid.cuboid.centroid
            size = self._grid.cuboid.size
            for ax in range(points.dim):
                if self.parameters["periodic"][ax]:
                    continue  # do nothing for periodic axes
                dist_norm = (pos[..., ax] - midpoint[ax]) / size[ax]
                factor = np.sign((dist_norm - 0.5) % 2 - 1)
                factor[factor == 0] = 1  # don't flip corner cases
                points.data[..., points.dim + ax] *= factor

        # move the points to inside the box
        pos[...] = self._grid.normalize_point(pos, reflect=True)
