"""
Provides a class representing the full system state of multiple elements

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
import logging
from typing import Any, Dict, Union, Sequence, Tuple, Optional  # @UnusedImport
from collections import defaultdict, OrderedDict

from pde.grids.base import DimensionError
from pde.tools.misc import hdf_write_attributes
from pde.tools.plotting import plot_on_axes

from .elements.base import ElementBase, element_from_hdf


class State:
    """ defines the state of the simulation """

    def __init__(self, elements: Dict[str, ElementBase] = None):
        """
        Args:
            elements (dict):
                Lists the elements in the simulation. The key in this dictionary
                gives the name of the element, while the associated value should
                be an instance of :class:`~sim.elements.base.ElementBase`.
        """
        self._logger = logging.getLogger(__name__)
        self.elements: Dict[str, ElementBase] = OrderedDict()
        self.dim: Optional[int] = None
        if elements:
            for name, element in elements.items():
                self.add_element(name, element)

    @classmethod
    def from_hdf_dataset(cls, dataset) -> "State":
        """ construct the instance by reading data from an hdf5 dataset
         
        Args:
            dataset: the hdf5 dataset (in an already opened file)
        """
        return cls(
            {
                name: element_from_hdf(dataset[name])
                for name in json.loads(dataset.attrs["elements"])
            }
        )

    @classmethod
    def from_file(cls, path: str) -> "State":
        """ create simulation state instance from data stored in a hdf file
          
        Args:
            path (str): Path to the hdf file being read
        """
        import h5py

        with h5py.File(path, "r") as fp:
            return cls.from_hdf_dataset(fp)

    def add_element(self, name: str, element: ElementBase):
        """ adds an element to the simulation
        
        Args:
            name (str):
                The identifier for the element.
            element (:class:`~sim.elements.base.ElementBase`):
                The instance defining the element.
        """
        if name in self.elements:
            self._logger.warning("Overwriting element `%s` in state", name)
        if len(self.elements) == 0:
            self.dim = element.dim
        elif self.dim != element.dim:
            raise DimensionError(
                f"Dimension of element ({element.dim}) differs "
                f"from state ({self.dim})"
            )
        self.elements[name] = element

    def get_index(self, name: str) -> int:
        """ returns the numerical index of a specific element
        
        Args:
            name (str): The name of the element        
        """
        for i, element_name in enumerate(self.elements):
            if name == element_name:
                return i
        raise KeyError(f"`{name}` not in {self.__class__.__name__}")

    def __getitem__(self, key: Union[str, Sequence[str]]):
        if isinstance(key, str):
            return self.elements[key]
        else:
            return [self.elements[k] for k in key]

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements.items())

    def __contains__(self, name: str):
        return name in self.elements

    def __str__(self):
        elements_str = ", ".join(f'"{name}": {element!s}' for name, element in self)
        return f"{self.__class__.__name__}({{{elements_str}}})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.elements!r})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(self.elements[key] == other.elements[key] for key in self.elements)

    def copy(self) -> "State":
        """ copy the state """
        return self.__class__({name: element.copy() for name, element in self})

    @property
    def attributes(self) -> Dict[str, Any]:
        """ dict: information about the state """
        return {"elements": {name: element.attributes for name, element in self}}

    def _write_hdf_dataset(self, hdf_path):
        """ write data to a given hdf5 file
        
        Args:
            dataset: the hdf5 dataset (in an already opened file)
         """
        element_names = []
        for name, element in self.elements.items():
            element_names.append(name)
            element._write_hdf_dataset(hdf_path.create_group(name))
        hdf_write_attributes(hdf_path, {"elements": element_names})

    def to_file(self, filename: str, info: Dict[str, Any] = None) -> None:
        r""" store elements in a hdf file
        
        Args:
            filename (str):
                Path where the data is stored
            info (dict):
                Extra information that is written to the hdf attributes. Note
                that the values in this dictionary will be JSON-serialized. 
        """
        import h5py

        with h5py.File(filename, "w") as fp:
            self._write_hdf_dataset(fp)
            hdf_write_attributes(fp, info)

    @property
    def data(self) -> Tuple[Any, ...]:
        """ tuple: the full data of the state  s"""
        return tuple(element.data for element in self.elements.values())

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom of the simulation """
        return sum(element.degrees_of_freedom for element in self.elements.values())

    @plot_on_axes()
    def plot(self, ax, element_args: Dict[str, Any] = None, **kwargs):
        r""" visualize the state
         
        Args:
            element_args (dict):
                A dictionary with arguments passed to the plotting functions of
                individual elements
            {PLOT_ARGS}
            **kwargs:
                All additional arguments are passed to all plotting functions
        """
        # prepare the element argument dict so it can be easily used below
        if element_args:
            element_args = defaultdict(dict, element_args)
        else:
            element_args = defaultdict(dict)

        # initialize the bounding box
        from matplotlib.transforms import Bbox

        limits = Bbox.null()

        # plot all elements individually
        for name, element in self:
            element.plot(ax=ax, **element_args[name], **kwargs)
            # keep track of the maximal bounding box
            limits.update_from_data_xy(ax.viewLim.get_points(), ignore=False)

        # set the bounding box to the maximal value
        ax.set_xlim(*limits.intervalx)
        ax.set_ylim(*limits.intervaly)
