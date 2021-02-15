"""
Provides an element that represents a collection of points

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Dict

import numpy as np

from pde.tools.parameters import Parameter
from pde.tools.plotting import plot_on_axes

from .base import ElementBase


class PointsElement(ElementBase):
    """ an element that represents a collection of points """

    parameters_default = [
        Parameter(
            "representative_radius",
            1,
            float,
            "Radius used for representing the point when plotting",
        )
    ]

    def __init__(self, data: np.ndarray = None, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`numpy.ndarray`):
                The positions of all points
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
        """
        # initialize parameters
        super().__init__(data, parameters)

        # ensure the right format of the input data
        self._data = np.atleast_2d(data)
        if self.data.ndim != 2:
            raise ValueError("`positions` must be a sequence of positions")
        self.dim = self.data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom for this element """
        return int(self.data.size)

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
            positions = np.c_[np.zeros(len(self)), self.data]
        elif self.dim == 2:
            positions = self.data
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
        result["data"] = self.data

        return result
