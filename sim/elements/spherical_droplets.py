"""
Provides a simulation element representing spherical droplets

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Dict, Any

import numpy as np

from pde.tools.parameters import Parameter
from droplets import SphericalDroplet, Emulsion

from .base import ElementBase


class SphericalDropletsElement(ElementBase):
    """ an element representing many droplets """

    droplet_class = SphericalDroplet

    parameters_default = [
        Parameter(
            "droplet_concentration",
            1,
            float,
            "Concentration inside droplets that is used to calculate the "
            "total amount of material in droplets",
        )
    ]

    def __init__(self, data: np.ndarray, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`numpy.ndarray`):
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

        super().__init__(None, parameters)
        self.droplets = Emulsion(
            [
                self.droplet_class.from_data(data_row)  # type: ignore
                for data_row in data
            ]
        )
        if len(self.droplets) == 0:
            raise ValueError(
                "At least a single droplet needs to be defined to "
                "determine the dimensionality of the element."
            )

        self.data = self.droplets.get_linked_data()
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
        # call the parent __init__
        ElementBase.__init__(obj, None, parameters=parameters)

        # initialize droplets
        obj.droplets = Emulsion(droplets, copy=copy)
        for droplet in obj.droplets:
            if not isinstance(droplet, obj.droplet_class):
                cls_name = droplet.__class__.__name__
                raise ValueError(
                    "DropletAgentsElement does not support droplets "
                    f"of class `{cls_name}`"
                )

        obj.data = obj.droplets.get_linked_data()
        obj.dim = obj.droplets.dim

        return obj  # type: ignore

    def __len__(self) -> int:
        return len(self.droplets)

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom for this element """
        entries_per_droplet = np.r_[self.data[0].tolist()]
        return len(self.data) * len(entries_per_droplet)

    @property
    def droplet_count(self) -> int:
        """ int: the number of droplets in the emulsion
        
        This only counts droplets with non-zero radius.
        """
        return sum(droplet.radius > 0 for droplet in self.droplets)

    @property
    def total_amount(self) -> float:
        """ float: total amount in the droplets """
        total_volume = sum(droplet.volume for droplet in self.droplets)
        return float(self.parameters["droplet_concentration"]) * total_volume

    def plot(self, ax=None, *args, **kwargs):
        """ plot all droplets of this element
        
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
