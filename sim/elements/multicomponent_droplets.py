"""
Provides a simulation element representing spherical droplets

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numba.extending import register_jitable

from droplets import Emulsion, SphericalDroplet
from droplets.tools.spherical import volume_from_radius
from pde.grids.base import GridBase

from .base import ElementBase


class MulticomponentDroplet(SphericalDroplet):
    """represents a single droplet comprised of many species"""

    __slots__ = ["data"]

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        amounts: np.ndarray,
    ):
        """
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            amounts (:class:`~numpy.ndarray`):
                The amounts of material of each component
        """
        self._init_data(position=position, amounts=amounts)
        self.data["amounts"] = amounts
        super().__init__(position=position, radius=radius)

    @classmethod
    def from_composition(
        cls,
        position: np.ndarray,
        radius: float,
        phis: np.ndarray,
    ) -> MulticomponentDroplet:
        """
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            phis (:class:`~numpy.ndarray`):
                The composition of the droplet
        """
        # TODO: Add optional field that is used to subtract background
        dim = len(position)
        amounts = np.asarray(phis) * volume_from_radius(radius, dim)
        return cls(position, radius, amounts)

    @classmethod
    def get_dtype(cls, **kwargs):
        """determine the dtype representing this droplet class

        Args:
            position (:class:`~numpy.ndarray`):
                The position vector of the droplet. This is used to determine the space
                dimension.
            amounts (:class:`~numpy.ndarray`):
                The amounts of material of each component

        Returns:
            :class:`numpy.dtype`: the (structured) dtype associated with this class
        """
        # extract data
        amounts = kwargs.pop("amounts")
        if amounts is None:
            assert TypeError("Require `amounts` argument")
        num_comps = len(amounts)
        assert num_comps >= 1

        # create dtype
        dtype = super().get_dtype(**kwargs)
        return dtype + [("amounts", float, (num_comps,))]

    @property
    def num_comps(self) -> int:
        """int: number of components inside the droplet"""
        shape = self.data.dtype.fields["amounts"][0].shape
        return int(shape[0]) if shape else 1

    @property
    def data_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """tuple: lower and upper bounds on the parameters"""
        l, h = super().data_bounds
        n = self.dim + 2
        l[n : n + self.num_comps] = 0
        return l, h

    def check_data(self):
        """method that checks the validity and consistency of self.data"""
        super().check_data()
        if np.any(self.amounts < 0):
            raise ValueError(f"Amounts must be positive ({self.amounts})")

    @property
    def amounts(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the composition"""
        return self.data["amounts"]

    @amounts.setter
    def amounts(self, value: np.ndarray):
        self.data["amounts"] = np.broadcast_to(value, (self.num_comps,))
        self.check_data()

    @property
    def phis(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: total amounts in the droplet"""
        return self.amounts / self.volume

    @property
    def phi_solvent(self) -> float:
        """float: solvent fraction"""
        return float(1 - self.phis.sum())

    @classmethod
    def _make_merge_data(cls) -> Callable[[np.ndarray, np.ndarray, np.ndarray], None]:
        """factory for a function that merges the data of two droplets"""
        parent_merge = super()._make_merge_data()

        @register_jitable
        def merge_data(drop1: np.ndarray, drop2: np.ndarray, out: np.ndarray) -> None:
            """merge the data of two droplets"""
            parent_merge(drop1, drop2, out)
            out.amounts[...] = drop1.amounts + drop2.amounts  # type: ignore

        return merge_data  # type: ignore

    def _get_phase_field(self, grid: GridBase, dtype=np.double) -> np.ndarray:
        """Creates a normalized image of the droplet on the `grid`

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field

        Returns:
            :class:`~numpy.ndarray`: An array with data values representing the droplet
            phase fields at support points of the `grid`.
        """
        phase_field = super()._get_phase_field(grid, dtype=float)
        return np.outer(self.phis, phase_field).astype(dtype)

    def _get_mpl_patch(self, dim=None, *, brightness=0.5, **kwargs):
        """return the patch representing the droplet for plotting

        The color of the droplets is determined automatically using at most the first
        three concentrations inside the droplet.

        Args:
            dim (int, optional):
                The dimension in which the data is plotted. If omitted, the actual
                physical dimension is assumed.
            brightness (float):
                Factor that determines how bright the colors of the droplets are
            **kwargs:
                Additional keyword arguments are passed to
                :class:`matplotlib.patches.Circle`, which creates the patch that
                represents the droplet. For instance, to only draw the outlines of the
                droplets, you may need to supply `fill=False`.

        Returns:
            :class:`~matplotlib.patches.Circle`: The patch representing the droplet
        """
        if kwargs.get("color") is None:
            color = np.ones(3)
            num_channels = min(3, self.num_comps)
            rgb = self.phis[:num_channels] * brightness * self.num_comps
            color[:num_channels] = np.clip(rgb, 0, 1)
            kwargs["color"] = color

        return super()._get_mpl_patch(dim=dim, **kwargs)


class MulticomponentDropletsElement(ElementBase):
    """an element representing many droplets"""

    _data: np.recarray
    dim: Optional[int]

    droplet_class = MulticomponentDroplet

    def __init__(self, data: np.ndarray, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`~numpy.ndarray`):
                The positions and radii of all points. This should be a
                structured array as returned by
                :attr:`~droplets.emulsions.Emulsion.data`
            parameters (dict):
                Additional parameters. Call
                :meth:`~sim.elements.spherical_droplets.SphericalDropletsElement.show_parameters`
                for details.
        """
        if isinstance(data, Emulsion):
            raise TypeError(
                "`data` should be a numpy array, not `Emulsion`. To initialize "
                f"`{self.__class__.__name__}` with an emulsions use "
                "the `from_droplets` classmethod."
            )
        if (
            hasattr(data, "__iter__")
            and len(data) > 0
            and isinstance(data[0], self.droplet_class)
        ):
            raise TypeError(
                "`data` should only contain the droplet data, no objects. To "
                f"initialize `{self.__class__.__name__}` with an emulsions use "
                "the `from_droplets` classmethod."
            )

        # set temporary data first and overwrite it later
        super().__init__(None, parameters)
        droplets = [self.droplet_class.from_data(data_row) for data_row in data]
        self.droplets = Emulsion(droplets)  # type: ignore

        if len(self.droplets) == 0:
            self._data = data.copy()
            self.dim = None
        else:
            self._data = self.droplets.get_linked_data()  # type: ignore
            self.dim = self.droplets.dim

    @classmethod
    def from_droplets(
        cls, droplets: Emulsion, copy: bool = False, parameters: Dict[str, Any] = None
    ) -> MulticomponentDropletsElement:
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
                raise ValueError(f"{cls.__name__} does not support `{cls_name}`")

        obj._data = obj.droplets.get_linked_data()  # type: ignore
        obj.dim = obj.droplets.dim

        return obj

    def __len__(self) -> int:
        return len(self.droplets)

    @property
    def droplet_count(self) -> int:
        """int: the number of droplets in the emulsion

        This only counts droplets with non-zero radius.
        """
        return sum(droplet.radius > 0 for droplet in self.droplets)

    @property
    def num_comps(self) -> Optional[int]:
        """int: the number of components inside each droplet"""
        if len(self) > 0:
            return int(self.droplets[0].num_comps)
        else:
            return None

    @property
    def amounts(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the amounts in all droplets"""
        return sum(droplet.amounts for droplet in self.droplets)  # type: ignore

    @property
    def total_amount(self) -> float:
        """float: total amount of all fields in all droplets"""
        return sum(self.amounts)  # type: ignore

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
