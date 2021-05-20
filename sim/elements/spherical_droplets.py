"""
Provides a simulation element representing spherical droplets

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Dict

import numpy as np

from droplets import Emulsion, SphericalDroplet
from pde.tools.parameters import Parameter

from .base import ElementBase


class SphericalDropletsElement(ElementBase):
    """ an element representing many droplets """

    parameters_default = [
        Parameter(
            "droplet_concentration",
            1,
            float,
            "Concentration inside droplets that is used to calculate the total amount "
            "of material in droplets",
        )
    ]

    droplet_class = SphericalDroplet

    _data: np.recarray

    def __init__(self, data: np.ndarray, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`~numpy.ndarray`):
                The positions and radii of all points. This should be a
                structured array as returned by
                :attr:`droplets.emulsions.Emulsion.data`
            parameters (dict):
                Additional parameters. Call
                :meth:`~SphericalDropletsElement.show_parameters` for details.
        """
        if isinstance(data, Emulsion) or isinstance(data[0], SphericalDroplet):
            raise TypeError(
                "`data` should be a numpy array. To initialize "
                f"`{self.__class__.__name__}` with an emulsions use "
                "the `from_droplets` classmethod."
            )

        # set temporary data first and overwrite it later
        super().__init__(None, parameters)
        droplets = [self.droplet_class.from_data(data_row) for data_row in data]
        self.droplets = Emulsion(droplets)  # type: ignore
        if len(self.droplets) == 0:
            raise ValueError(
                "At least a single droplet needs to be defined to "
                "determine the dimensionality of the element."
            )

        self._data = self.droplets.get_linked_data()
        self.dim = self.droplets.dim  # type: ignore

    @classmethod
    def from_droplets(
        cls, droplets: Emulsion, copy: bool = False, parameters: Dict[str, Any] = None
    ) -> "SphericalDropletsElement":
        """
        Args:
            droplets (:class:`droplets.emulsions.Emulsion`):
                The state of this element given as an emulsion.
            copy (bool):
                Flag indicating whether the droplets are copied, so they are not
                modified during the simulation.
            parameters (dict):
                Additional parameters. Call
                :meth:`~SphericalDropletsElement.show_parameters` for details.
        """
        # create class without calling its __init__
        obj = cls.__new__(cls)
        # call the parent __init__ with a temporary array
        ElementBase.__init__(obj, None, parameters=parameters)

        # initialize droplets
        obj.droplets = Emulsion(droplets, copy=copy)
        for droplet in obj.droplets:
            if not isinstance(droplet, obj.droplet_class):
                cls_name = droplet.__class__.__name__
                raise ValueError(f"DropletAgentsElement does not support `{cls_name}`")

        obj._data = obj.droplets.get_linked_data()
        obj.dim = obj.droplets.dim

        return obj  # type: ignore

    def __len__(self) -> int:
        return len(self.droplets)

    @property
    def droplet_count(self) -> int:
        """int: the number of droplets in the emulsion

        This only counts droplets with non-zero radius.
        """
        return sum(droplet.radius > 0 for droplet in self.droplets)

    @property
    def total_amount(self) -> float:
        """ float: total amount in the droplets """
        total_volume = sum(droplet.volume for droplet in self.droplets)
        return float(self.parameters["droplet_concentration"]) * total_volume

    def plot(self, ax=None, *args, **kwargs):
        """plot all droplets of this element

        Args:
            {PLOT_ARGS}
            **kwargs:
                All additional arguments are forwarded to
                :meth:`droplets.emulsions.Emulsion.plot`.
        """
        emulsion = self.droplets
        if "grid" in kwargs:
            emulsion = emulsion.copy()
            emulsion.grid = kwargs.pop("grid")
        emulsion.plot(ax=ax, *args, **kwargs)

    def _get_napari_layer_data(
        self, point_like: bool = False, resolution: float = 1, **kwargs
    ) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

        Args:
            point_like (bool):
                Flag indicating whether droplets are rendered as points or as extended
                shapes
            resolution (float):
                The typical length of the discretized representation. This argument is
                only used when `point_like` is `False`.
            **kwargs:
                Additional arguments returned in the result, which affect how the layer
                is shown.

        Returns:
            dict: all the information necessary to plot the points
        """
        result = kwargs

        if point_like:
            # render droplets as points
            result.setdefault("size", self.data["radius"])
            result.setdefault("n_dimensional", True)  # render point in all layers
            result["type"] = "points"
            result["data"] = self.data["position"]

        elif self.dim == 2:
            # render 2d droplets as (closed) paths
            result.setdefault("edge_width", 0.1)
            data = [
                droplet.get_triangulation(resolution=resolution)["vertices"]
                for droplet in self.droplets
            ]

            result["type"] = "shapes"
            result["shape_type"] = "path"
            result["data"] = data

        elif self.dim == 3:
            # render 3d droplets as (closed) surfaces
            vertices = np.empty((0, 3), np.double)
            faces = np.empty((0, 3), np.int64)
            for droplet in self.droplets:
                tri = droplet.get_triangulation(resolution=resolution)
                offset = len(vertices)
                vertices = np.r_[vertices, tri["vertices"]]
                faces = np.r_[faces, tri["triangles"] + offset]

            result["type"] = "surface"
            result["data"] = (vertices, faces, np.ones(len(vertices)))

        return result
