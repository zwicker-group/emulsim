"""
Provides a simulation element representing spherical droplets

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from droplets import Emulsion, SphericalDroplet
from modelrunner.state import NoData
from pde.fields import FieldBase
from pde.grids.base import GridBase

from .. import Parameter
from .base import ArrayElementBase
from .fields import FieldElementBase


class SphericalDropletsElement(ArrayElementBase):
    """an element representing many droplets"""

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

    data: np.recarray

    def _state_init(self, attributes: Dict[str, Any], data=NoData) -> None:
        """initialize the state with attributes and (optionally) data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        super()._state_init(attributes, data)

        if data is not NoData:
            # extract the droplets from data if given
            drop_list = [
                self.droplet_class.from_data(data_row) for data_row in self.data
            ]
            droplets = Emulsion(
                drop_list, copy=False, dtype=self.data.dtype, force_consistency=True  # type: ignore
            )
            # initialize the droplet information in the class
            self._init_droplets(droplets)

    def _init_droplets(
        self,
        droplets: Emulsion,
        *,
        maxcount: Optional[int] = None,
    ):
        """helper function that ensures that the droplet attribute is linked with `data`

        Args:
            droplets:
                The droplet data in form of :class:`~droplets.emulsions.Emulsion`
            maxcount (int):
                If supplied, sets the maximal number of droplets that can be stored in
                this element. If `maxcount > len(droplets)`, the additional entries are
                intialized as zero.
        """
        if droplets.dtype is None:
            raise ValueError(
                "Need a valid emulsion to initialize element, i.e., there must be one "
                "droplet (with potentially vanishing radius) or dtype must be set."
            )

        if maxcount is not None:
            # ensure that len(droplets) >= maxcount
            if maxcount > len(droplets):
                # append empty droplets to list
                empty_droplet = self.droplet_class.from_data(
                    np.zeros(1, dtype=droplets.dtype)[0]
                )
                for _ in range(maxcount - len(droplets)):
                    droplets.append(empty_droplet, copy=True)  # type: ignore
                assert len(droplets) == maxcount

        self.droplets = droplets  # set the given droplets as an attribute

        # check whether all droplets are derived from the same class
        for droplet in droplets:
            if not isinstance(droplet, self.droplet_class):
                cls_name = droplet.__class__.__name__
                raise ValueError(f"`{cls_name}` is not `{self.droplet_class}`")

        # set additional information about droplets
        self.dim = self.droplets.dim
        self.data = self.droplets.get_linked_data().view(np.recarray)
        # note that the conversion to a record array is necessary so the attribute
        # access works when code intended for numba is executed in pure python, e.g.,
        # for coverage determination

    @classmethod
    def empty(
        cls,
        droplet: SphericalDroplet,
        maxcount: int,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SphericalDropletsElement:
        """create empty SphericalDropletsElement that can be filled with droplets later

        Args:
            droplet (:class:`~droplets.droplets.SphericalDroplet`):
                Example of a droplet to define the kind of emulsion
            maxcount (int):
                Sets the maximal number of droplets that can be stored in this element.
            parameters (dict):
                Additional parameters. Call
                :meth:`~SphericalDropletsElement.show_parameters` for details.

        Returns:
            :class:`SphericalDropletsElement`: The initialized element
        """
        obj = cls.__new__(cls)  # create class without calling its __init__
        obj._state_init({"parameters": parameters})  # initialize generally
        obj._init_droplets(
            Emulsion([], copy=False, dtype=droplet.data.dtype), maxcount=maxcount
        )
        return obj

    @classmethod
    def from_droplets(
        cls,
        droplets: Emulsion,
        *,
        copy: bool = False,
        maxcount: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SphericalDropletsElement:
        """
        Create `SphericalDropletsElement` from a list of droplets

        Args:
            droplets (:class:`droplets.emulsions.Emulsion`):
                The state of this element given as an emulsion.
            copy (bool):
                Flag indicating whether the droplets are copied, so they are not
                modified during the simulation.
            maxcount (int):
                If supplied, sets the maximal number of droplets that can be stored in
                this element. If `maxcount > len(droplets)`, the additional entries are
                intialized as zero.
            parameters (dict):
                Additional parameters. Call
                :meth:`~SphericalDropletsElement.show_parameters` for details.
        """
        obj = cls.__new__(cls)  # create class without calling its __init__
        obj._state_init({"parameters": parameters})  # initialize generally
        dtype = droplets.data.dtype if hasattr(droplets, "data") else None
        obj._init_droplets(
            Emulsion(droplets, copy=copy, dtype=dtype, force_consistency=True),
            maxcount=maxcount,
        )
        return obj

    @classmethod
    def from_random(
        cls,
        num: int,
        bounds: Union[FieldElementBase, FieldBase, GridBase],
        radius: Union[float, Tuple[float, float]],
        *,
        remove_overlapping: bool = True,
        maxcount: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SphericalDropletsElement:
        """
        Create `SphericalDropletsElement` with random droplets

        Args:
            num (int):
                The number of droplets
            bounds:
                Boundaries of the space in which droplets are placed. This can be any of
                :class:`FieldElementBase`, :class:`FieldBase`, or :class:`GridBase`.
            radius (float or tuple of float):
                Radius of the droplets that are created. If two numbers are given, they
                specify the bounds of a uniform distribution from which the radius of
                each individual droplet is chosen.
            remove_overlapping (bool):
                Flag determining whether overlapping droplets are removed. If enabled,
                the resulting element might contain less thatn `num` droplets.
            maxcount (int):
                If supplied, sets the maximal number of droplets that can be stored in
                this element. If `maxcount > num`, the additional entries are
                intialized as zero.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
            parameters (dict):
                Additional parameters. Call
                :meth:`~SphericalDropletsElement.show_parameters` for details.
        """
        # extract information about the bounds of the space
        if hasattr(bounds, "bounds"):
            axes_bounds = bounds.bounds
        elif hasattr(bounds, "axes_bounds"):
            axes_bounds = bounds.axes_bounds
        else:
            axes_bounds = np.atleast_2d(bounds)  # type: ignore

        # create a random emulsion
        emulsion = Emulsion.from_random(
            num=num,
            grid_or_bounds=axes_bounds,
            radius=radius,
            remove_overlapping=remove_overlapping,
            droplet_class=cls.droplet_class,
            rng=rng,
        )

        # turn the emulsion into a SphericalDropletsElement
        return cls.from_droplets(
            emulsion,
            copy=False,
            maxcount=maxcount,
            parameters=parameters,
        )

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
        """float: total amount in the droplets"""
        total_volume = sum(droplet.volume for droplet in self.droplets)
        return float(self.parameters["droplet_concentration"]) * total_volume  # type: ignore

    def plot(self, ax=None, *args, **kwargs):
        """plot all droplets of this element

        Args:
            {PLOT_ARGS}
            **kwargs:
                All additional arguments are forwarded to
                :meth:`droplets.emulsions.Emulsion.plot`.
        """
        plot_args = self.parameters["plot_args"].copy()
        plot_args.update(kwargs)

        emulsion = self.droplets
        if "grid" in kwargs:
            emulsion = emulsion.copy()
            emulsion.grid = kwargs.pop("grid")
        emulsion.plot(ax=ax, *args, **plot_args)

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
            faces = np.empty((0, 3), np.uint)
            for droplet in self.droplets:
                tri = droplet.get_triangulation(resolution=resolution)
                offset = len(vertices)
                vertices = np.r_[vertices, tri["vertices"]]
                faces = np.r_[faces, tri["triangles"] + offset]

            result["type"] = "surface"
            result["data"] = (vertices, faces)  # , np.ones(len(vertices)))

        return result
