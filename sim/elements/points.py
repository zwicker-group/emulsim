"""
Provides an element that represents a collection of points

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from typing import Any, Dict, Union

import numpy as np

from pde.tools.parameters import Parameter
from pde.tools.plotting import plot_on_axes

from .base import ElementBase


class PointsElement(ElementBase):
    """an element that represents a collection of points"""

    parameters_default = [
        Parameter(
            "representative_radius",
            1,
            float,
            "Radius used for representing the point when plotting",
        )
    ]

    def __init__(self, data: np.ndarray, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`~numpy.ndarray`):
                The positions of all points
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        data = np.asanyarray(data)

        # ensure the right format of the input data
        if data.dtype.fields:
            # record dtype
            self._logger.debug("Data of PointsElement was recarray")
            if data.ndim != 1 or "position" not in data.dtype.fields:
                raise ValueError(
                    "`data` must be an array of records with a `position` field"
                )
            self.dim = data.dtype["position"].shape[0]
            if isinstance(data, np.recarray):
                rec_data = data
            else:
                rec_data = data.view(np.recarray)

        else:
            # simple dtype
            self._logger.info("Data of PointsElement needs to be promoted to recarray")
            data = np.atleast_2d(data)
            if data.ndim != 2:
                raise ValueError("`data` must be a sequence of positions")

            num_el, self.dim = data.shape
            rec_data = np.recarray((num_el,), dtype=[("position", float, (self.dim,))])
            rec_data.position[:] = data

        # initialize parameters
        super().__init__(rec_data, parameters)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def positions(self) -> np.ndarray:
        return self.data["position"]  # type: ignore

    @positions.setter
    def positions(self, value: np.ndarray) -> None:
        self.data["position"] = value

    @plot_on_axes()
    def plot(self, ax, color="red", **kwargs):
        """plot all points of this element

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
        radius = self.parameters["representative_radius"]
        patches = [mpl.patches.Circle(pos, radius) for pos in positions]

        # add all patches as a collection
        coll = mpl.collections.PatchCollection(patches, facecolors=(color,))
        ax.add_collection(coll)

        # determine bounding box
        xmin, ymin = positions.min(axis=0) - radius
        xmax, ymax = positions.max(axis=0) + radius
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

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
    """an element that represents a collection of points with direction"""

    #
    # parameters_default = [
    #     Parameter(
    #         "representative_radius",
    #         1,
    #         float,
    #         "Radius used for representing the point when plotting",
    #     )
    # ]

    def __init__(self, data: np.recarray, parameters: Dict[str, Any] = None):
        """
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
        # initialize parameters
        super().__init__(data, parameters)
        #
        # if not isinstance(self.data, np.recarray) or self.data.ndim != 1:
        #     raise ValueError(
        #         "`data` must be a record array with fields `position` and `direction` "
        #         "specifying these for all points."
        #     )
        # dtype = self.data.dtype
        # self.dim = dtype["position"].shape[0]
        assert self.data.dtype["direction"].shape == (self.dim,)

    @classmethod
    def from_position_direction(
        cls,
        positions: np.ndarray,
        directions: np.ndarray,
        parameters: Dict[str, Any] = None,
    ) -> "ArrowsElement":
        """create element from separately specified positions and directions

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
        direction_magnitude: Union[float, np.ndarray] = 1,
        parameters: Dict[str, Any] = None,
    ) -> "ArrowsElement":
        """create element from separately specified positions and directions

        Args:
            positions (:class:`~numpy.ndarray`):
                The positions of all points
            directions (float or :class:`~numpy.ndarray`):
                The magnitude of the direction vector. Either a single number or an
                array specifying values for each point can be given
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
        """
        positions = np.atleast_2d(positions)
        num_points, dim = positions.shape
        magnitude = np.array(direction_magnitude, np.double, ndmin=1)

        if dim == 1:
            directions = magnitude * np.random.choice([-1.0, 1.0], size=num_points)
            directions = directions.reshape(-1, 1)
        elif dim == 2:
            φs = np.random.uniform(0, 2 * np.pi, size=num_points)
            directions = magnitude[:, np.newaxis] * np.c_[np.sin(φs), np.cos(φs)]
        else:
            raise NotImplementedError

        return cls.from_position_direction(positions, directions, parameters)

    def __len__(self) -> int:
        return len(self.data)

    #
    # @property
    # def positions(self) -> np.ndarray:
    #     return self.data.position  # type: ignore
    #
    # @positions.setter
    # def positions(self, value: np.ndarray) -> None:
    #     self.data.position = value

    @property
    def directions(self) -> np.ndarray:
        return self.data["direction"]  # type: ignore

    @directions.setter
    def directions(self, value: np.ndarray) -> None:
        self.data["direction"] = value

    #
    # @plot_on_axes()
    # def plot(self, ax, color="red", **kwargs):
    #     """plot all points of this element
    #
    #     Args:
    #         color (matplotlib color):
    #             The color with which the points are shown
    #         {PLOT_ARGS}
    #     """
    #     import matplotlib as mpl
    #
    #     if self.dim == 1:
    #         positions = np.c_[np.zeros(len(self)), self.positions]
    #     elif self.dim == 2:
    #         positions = self.positions
    #     else:
    #         raise RuntimeError(f"Cannot plot points with dimension {self.dim}")
    #
    #     # create the patches
    #     radius = self.parameters["representative_radius"]
    #     patches = [mpl.patches.Circle(pos, radius) for pos in positions]
    #
    #     # add all patches as a collection
    #     # TODO represent data by arrows
    #     coll = mpl.collections.PatchCollection(patches, facecolors=(color,))
    #     ax.add_collection(coll)
    #
    #     # determine bounding box
    #     xmin, ymin = positions.min(axis=0) - radius
    #     xmax, ymax = positions.max(axis=0) + radius
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #
    # def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
    #     """returns data for plotting on a single napari layer
    #
    #     Args:
    #         size (float):
    #             The size of the points
    #         **kwargs:
    #             Additional arguments returned in the result, which affect how the layer
    #             is shown.
    #
    #     Returns:
    #         dict: all the information necessary to plot the points
    #     """
    #     result = kwargs
    #
    #     # TODO represent data by arrows
    #     result.setdefault("size", 1)
    #     result["type"] = "points"
    #     result["data"] = self.positions
    #
    #     return result
