"""Provides an element that represents a collection of points.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pde.tools.plotting import plot_on_axes

from .. import Parameter
from .base import ArrayElementBase, NoData


class PointsElement(ArrayElementBase):
    """Element representing a collection of points."""

    parameters_default = [
        Parameter(
            "plot_radius",
            1,
            float,
            "Radius used for representing the point when plotting",
        )
    ]

    def _init_state(self, attributes: dict[str, Any], data=NoData) -> None:
        """Initialize the state with attributes and (optionally) data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degrees of freedom of the physical system
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        # data = np.asanyarray(data)

        super()._init_state(attributes, data)

        # ensure the right format of the input data
        if self.data.dtype.fields:
            # record dtype
            self._logger.debug("Data of PointsElement was recarray")
            if self.data.ndim != 1 or "position" not in self.data.dtype.fields:
                raise ValueError("`data` must be recarray with a `position` field")
            self.dim = self.data.dtype["position"].shape[0]
            self.data = self.data.view(np.recarray)

        else:
            # simple dtype
            self._logger.info("Data of PointsElement needs to be promoted to recarray")
            data = np.atleast_2d(self.data)
            if data.ndim != 2:
                raise ValueError("`data` must be a sequence of positions")

            num_el, self.dim = data.shape
            self.data = np.recarray((num_el,), dtype=[("position", float, (self.dim,))])
            self.data.position[:] = data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def positions(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the positions of all points"""
        return self.data["position"]  # type: ignore

    @positions.setter
    def positions(self, value: np.ndarray) -> None:
        self.data["position"] = value

    @plot_on_axes()
    def plot(self, ax, color="red", **kwargs):
        """Plot all points of this element.

        Args:
            color (matplotlib color):
                The color with which the points are shown
            {PLOT_ARGS}
        """
        import matplotlib as mpl

        if self.dim == 1:
            positions = np.c_[np.zeros(len(self)), self.positions]
        elif self.dim == 2:
            positions = self.positions
        else:
            raise RuntimeError(f"Cannot plot points with dimension {self.dim}")

        # create the patches
        radius = self.parameters["plot_radius"]
        patches = [mpl.patches.Circle(pos, radius) for pos in positions]

        # add all patches as a collection
        plot_args = self.parameters["plot_args"].copy()
        plot_args.update(kwargs)
        plot_args.setdefault("facecolors", (color,))
        coll = mpl.collections.PatchCollection(patches, **plot_args)
        ax.add_collection(coll)

        # determine bounding box
        xmin, ymin = positions.min(axis=0) - radius
        xmax, ymax = positions.max(axis=0) + radius
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _get_napari_layer_data(self, **kwargs) -> dict[str, Any]:
        """Returns data for plotting on a single napari layer.

        Args:
            size (float):
                The size of the points
            **kwargs:
                Additional arguments returned in the result, which affect how the layer
                is shown.

        Returns:
            dict: all the information necessary to plot the points
        """
        result = kwargs

        result.setdefault("size", 1)
        result["type"] = "points"
        result["data"] = self.positions

        return result


class ArrowsElement(PointsElement):
    """Element representing a collection of points with direction.

    Args:
        data (:class:`~numpy.recarray`):
            The structured array with entries for 'position' and 'direction' for all
            points. For example, the dtype of the array should be
            `[("position", float, (dim,)), ("direction", float, (dim,))]`, where
            `dim` is the dimension of space.
        parameters (dict):
            Additional parameters. Call
            :meth:`~PointsElement.show_parameters` for details.
    """

    def _init_state(self, attributes: dict[str, Any], data=NoData) -> None:
        """Initialize the state with attributes and (optionally) data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degrees of freedom of the physical system
        """
        super()._init_state(attributes, data)
        dir_shape = self.data.dtype["direction"].shape
        if dir_shape != (self.dim,):
            raise ValueError(f"Direction must have shape {(self.dim)}, got {dir_shape}")

    @classmethod
    def from_position_direction(
        cls,
        positions: np.ndarray,
        directions: np.ndarray,
        parameters: dict[str, Any] | None = None,
    ) -> ArrowsElement:
        """Create element from separately specified positions and directions.

        Args:
            positions (:class:`~numpy.ndarray`):
                The positions of all points
            directions (:class:`~numpy.ndarray`):
                The directions of all points
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
        """
        positions, directions = np.broadcast_arrays(positions, directions)
        num_el, dim = positions.shape

        dtype = [("position", float, (dim,)), ("direction", float, (dim,))]
        data: np.recarray = np.recarray((num_el,), dtype=dtype)
        data.position = positions
        data.direction = directions

        return cls(data, parameters)

    @classmethod
    def from_position_random_direction(
        cls,
        positions: np.ndarray,
        direction_magnitude: float | np.ndarray = 1,
        parameters: dict[str, Any] | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> ArrowsElement:
        """Create element from separately specified positions and directions.

        Args:
            positions (:class:`~numpy.ndarray`):
                The positions of all points
            directions (float or :class:`~numpy.ndarray`):
                The magnitude of the direction vector. Either a single number or an
                array specifying values for each point can be given
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        rng = np.random.default_rng(rng)
        positions = np.atleast_2d(positions)
        num_points, dim = positions.shape
        magnitude: np.ndarray = np.array(direction_magnitude, np.double, ndmin=1)

        if dim == 1:
            directions = magnitude * rng.choice([-1.0, 1.0], size=num_points)
            directions = directions.reshape(-1, 1)
        elif dim == 2:
            φs = rng.uniform(0, 2 * np.pi, size=num_points)
            directions = magnitude[:, np.newaxis] * np.c_[np.sin(φs), np.cos(φs)]
        else:
            raise NotImplementedError

        return cls.from_position_direction(positions, directions, parameters)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def directions(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the directions of all arrows"""
        return self.data["direction"]  # type: ignore

    @directions.setter
    def directions(self, value: np.ndarray) -> None:
        self.data["direction"] = value

    @plot_on_axes()
    def plot(self, ax, color="red", **kwargs):
        """Plot all points of this element.

        Args:
            color (matplotlib color):
                The color with which the points are shown
            {PLOT_ARGS}
        """
        import matplotlib as mpl

        if self.dim == 1:
            positions = np.c_[np.zeros(len(self)), self.positions]
        elif self.dim == 2:
            positions = self.positions
        else:
            raise RuntimeError(f"Cannot plot points with dimension {self.dim}")

        # create the patches
        radius = self.parameters["plot_radius"]
        patches = [mpl.patches.Circle(pos, radius) for pos in positions]

        # add all patches as a collection
        # TODO represent data by arrows
        plot_args = self.parameters["plot_args"].copy()
        plot_args.update(kwargs)
        plot_args.setdefault("facecolors", (color,))
        coll = mpl.collections.PatchCollection(patches, **plot_args)
        ax.add_collection(coll)

        # determine bounding box
        xmin, ymin = positions.min(axis=0) - radius
        xmax, ymax = positions.max(axis=0) + radius
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
