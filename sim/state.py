"""
Provides a class representing the full system state of multiple elements

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy
import itertools
import json
import logging
from collections import OrderedDict, defaultdict
from typing import Optional  # @UnusedImport
from typing import Any, Dict, Iterable, Sequence, Set, Tuple, Union

from pde.grids.base import DimensionError, GridBase
from pde.tools.misc import hdf_write_attributes
from pde.tools.parameters import Parameter, Parameterized
from pde.tools.plotting import napari_add_layers, plot_on_axes

from .elements.base import ElementBase


class State(Parameterized):
    """defines the state of the simulation as a collection of elements"""

    parameters_default = [
        Parameter(
            "bounds",
            None,
            object,
            "Bounds of the simulation box, which affects plotting",
        ),
        Parameter(
            "invisible_elements",
            set(),
            set,
            "Collection of elements that will not be plotted",
        ),
    ]

    def __init__(
        self, elements: Dict[str, ElementBase] = None, parameters: Dict[str, Any] = None
    ):
        """
        Args:
            elements (dict):
                Lists the elements in the simulation. The key in this dictionary
                gives the name of the element, while the associated value should
                be an instance of :class:`~sim.elements.base.ElementBase`.
            parameters (dict):
                Parameters that affect the entire state
        """
        super().__init__(parameters)
        self._logger = logging.getLogger(__name__)

        # determine dimensionality of space
        if self.parameters["bounds"] is not None:
            self.dim: Optional[int] = len(self.parameters["bounds"])
        else:
            self.dim = None

        # add elements to the simulation
        self.elements: Dict[str, ElementBase] = OrderedDict()
        if elements:
            for name, element in elements.items():
                self.add_element(name, element)

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> "State":
        """construct the instance by reading data from an hdf5 dataset

        Args:
            dataset: the hdf5 dataset (in an already opened file)
        """
        element_names = json.loads(dataset.attrs["elements"])
        elements = {
            name: ElementBase._from_hdf_dataset(dataset[name]) for name in element_names
        }
        return cls(elements)

    @classmethod
    def from_file(cls, path: str) -> "State":
        """create simulation state instance from data stored in a hdf file

        Args:
            path (str): Path to the hdf file being read
        """
        import h5py

        with h5py.File(path, "r") as fp:
            return cls._from_hdf_dataset(fp)

    def add_element(self, name: str, element: ElementBase):
        """adds an element to the simulation

        Args:
            name (str):
                The identifier for the element.
            element (:class:`~sim.elements.base.ElementBase`):
                The instance defining the element.
        """
        if name in self.elements:
            self._logger.warning("Overwriting element `%s` in state", name)

        # check dimensionality
        if element.dim is None:
            pass
        elif self.dim is None:
            self.dim = element.dim
        elif self.dim != element.dim:
            raise DimensionError(
                f"Element dimension ({element.dim}) differs from state ({self.dim})"
            )
        self.elements[name] = element

    def get_index(self, name: str) -> int:
        """returns the numerical index of a specific element

        Args:
            name (str): The name of the element
        """
        for i, element_name in enumerate(self.elements):
            if name == element_name:
                return i
        raise KeyError(f"`{name}` not in {self.__class__.__name__}")

    def __getitem__(self, key: Union[int, str, Sequence[str]]):
        """extract element by numerical index or by name"""
        if isinstance(key, int):
            # handle numerical index
            size = len(self)
            if -size <= key < size:
                if key < 0:
                    key += size
                return next(itertools.islice(self.elements.values(), key, key + 1))
            else:
                raise IndexError("element index out of range")

        elif isinstance(key, str):
            # handle name index
            return self.elements[key]

        else:
            # handle multiple indices
            return tuple(self[k] for k in key)

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements.items())

    def __contains__(self, name: str):
        return name in self.elements

    def keys(self):
        return self.elements.keys()

    def values(self):
        return self.elements.values()

    def items(self):
        return self.elements.items()

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
        """copy the state"""
        return self.__class__(
            {name: element.copy() for name, element in self},
            parameters=copy.deepcopy(self.parameters),
        )

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the state"""
        return {
            "elements": {name: element.attributes for name, element in self},
            "parameters": self.parameters,
        }

    def _write_hdf_dataset(self, hdf_path):
        """write data to a given hdf5 file

        Args:
            dataset: the hdf5 dataset (in an already opened file)
        """
        element_names = []
        for name, element in self:
            element_names.append(name)
            element._write_hdf_dataset(hdf_path.create_group(name))
        hdf_write_attributes(hdf_path, {"elements": element_names})

    def to_file(self, filename: str, info: Dict[str, Any] = None) -> None:
        r"""store elements in a hdf file

        Args:
            filename (str):
                Path where the data is stored
            info (dict):
                Extra information that is written to the hdf attributes. Note
                that the values in this dictionary will be JSON-serialized.
        """
        import h5py

        if info is not None and "elements" in info:
            self._logger.warning("`elements` entry of `info` will be overwritten")

        with h5py.File(filename, "w") as fp:
            hdf_write_attributes(fp, info)
            self._write_hdf_dataset(fp)

    @property
    def data(self) -> Tuple[Any, ...]:
        """tuple: the full data of the state  s"""
        return tuple(element.data for element in self.elements.values())

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom of the simulation"""
        return sum(element.degrees_of_freedom for element in self.elements.values())

    def get_quantity(self, property_name: str, total: bool = True):
        """returns quantities obtained from the elements

        Quantities are typically implemented as properties or attributes. If
        an element does not have a property, it is silently ignored and not
        included in the result.

        Args:
            property_name (str):
                The name of the property or attribute that is analyzed
            total (bool):
                Flag determining whether the sum of all values is returned.

        Returns:
            float or dict: A total value is returned if total is `True`.
            Otherwise, the value for each element is returned in a dictionary.
        """
        if total:
            # return the sum over all properties
            return sum(
                getattr(element, property_name)
                for element in self.elements.values()
                if hasattr(element, property_name)
            )
        else:
            # return a dictionary with the quantity result
            result: Dict[str, Any] = {}
            for element_name, element in self:
                if hasattr(element, property_name):
                    result[element_name] = getattr(element, property_name)
            return result

    @plot_on_axes()
    def plot(
        self,
        ax,
        element_args: Dict[str, Any] = None,
        invisible_elements: Iterable[str] = None,
        **kwargs,
    ):
        r"""visualize the state

        Args:
            element_args (dict):
                A dictionary with arguments passed to the plotting functions of
                individual elements
            invisible_elements (list):
                A list of elements that will not be plotted.
            {PLOT_ARGS}
            **kwargs:
                All additional arguments are passed to all plotting functions
        """
        # prepare the element argument dict so it can be easily used below
        if element_args:
            element_args = defaultdict(dict, element_args)
        else:
            element_args = defaultdict(dict)

        if invisible_elements is None:
            ignore_el: Set[str] = self.parameters["invisible_elements"]
        else:
            ignore_el = set(invisible_elements) | self.parameters["invisible_elements"]

        # initialize the bounding box
        from matplotlib.transforms import Bbox

        limits = Bbox.null()

        # plot all elements individually
        for name, element in self:
            if name not in ignore_el:
                element.plot(ax=ax, **element_args[name], **kwargs)
                # keep track of the maximal bounding box
                limits.update_from_data_xy(ax.viewLim.get_points(), ignore=False)

        if self.parameters["bounds"] is None:
            # set the bounding box to the maximal value
            ax.set_xlim(*limits.intervalx)
            ax.set_ylim(*limits.intervaly)
        else:
            ax.set_xlim(*self.parameters["bounds"][0])
            ax.set_ylim(*self.parameters["bounds"][1])
            ax.set_aspect(1)

    def plot_interactive(
        self, grid: GridBase = None, viewer_args: Dict[str, Any] = None, **kwargs
    ):
        """create an interactive plot of the field using :mod:`napari`

        Args:
            grid (:~pde.grids.base.GridBase`):
                The grid that defines the space in which the simulation takes place. If
                omitted, we try to determine it automatically from the elements in the
                state.
            viewer_args (dict):
                Arguments passed to :class:`napari.viewer.Viewer` to affect the viewer
            **kwargs:
                Extra arguments passed to all plotting function
        """
        from pde.tools.plotting import napari_viewer

        if viewer_args is None:
            viewer_args = {}

        # try finding the best field that could serve to define the space
        layers_data = {}
        for name, element in self.elements.items():
            try:
                layer_data = element._get_napari_layer_data()
            except NotImplementedError:
                self._logger.warning(
                    "Element %s does not support interactive plotting", name
                )
            else:
                layers_data[name] = layer_data

            # try to find a suitable grid
            try:
                candidate = element.grid  # type: ignore
            except AttributeError:
                pass
            else:
                if isinstance(candidate, GridBase) and candidate.dim == self.dim:
                    if grid is None or candidate.volume > grid.volume:
                        grid = candidate

        # check whether we have enough information to proceed
        if grid is None:
            raise RuntimeError("Could not determine suitable grid")
        if grid.dim != self.dim:
            raise RuntimeError(
                "Grid dimension is not compatible (%d != %d)", grid.dim, self.dim
            )

        # do the actual plotting
        with napari_viewer(grid, **viewer_args) as viewer:
            napari_add_layers(viewer, layers_data)
