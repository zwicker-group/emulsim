"""Provides a class representing the full system state of multiple elements.

.. inheritance-diagram:: State
   :parts: 1


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import numpy as np

from modelrunner import Parameter, Parameterized
from pde.backends.numba.utils import jit
from pde.grids.base import DimensionError, GridBase
from pde.tools.plotting import napari_add_layers, plot_on_axes

from .elements.base import DictElementBase, NoData, _ElementBase


class State(DictElementBase):
    """Defines the state of the simulation as a collection of elements."""

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

    _check_dimension: bool = True
    _state_attributes_attr_name = "attributes"
    _state_data_attr_name = "data"

    data: dict[str, _ElementBase]

    def __init__(
        self,
        elements: dict[str, _ElementBase] | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Args:
            elements (dict):
                Lists the elements in the simulation. The key in this dictionary
                gives the name of the element, while the associated value should
                be an instance of :class:`~sim.elements.base._ElementBase`.
            parameters (dict):
                Parameters that affect the entire state
        """
        # parse parameters and initialize empty dictionary storage
        super().__init__({}, parameters)

        # determine dimensionality of space
        if self.parameters["bounds"] is None:
            self.dim: int | None = None  # cannot determine dimension at this point
        else:
            self.dim = len(self.parameters["bounds"])

        # add elements to the state
        if elements:
            for name, element in elements.items():
                self.add_element(name, element)

    def _init_state(self, attributes: dict[str, Any], data=NoData) -> None:
        """Initialize the state with attributes and (optionally) data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degrees of freedom of the physical system
        """
        if data is not NoData:
            self.data = data

        self.dim = attributes.pop("dim", None)
        self.parameters = self._parse_parameters(
            attributes["parameters"], include_deprecated=True, check_validity=True
        )
        if sum(1 for a in attributes if not a.startswith("_")) != 1:
            raise ValueError(f"Too many attributes: {attributes.keys()}")

    @property
    def _attributes_storage(self) -> dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage

        This property modifies the normal `_state_attributes` and adds information
        necessary for restoring the class using :meth:`StateBase.from_data`.
        """
        attrs = super()._attributes_storage
        attrs["dim"] = self.dim
        return attrs

    @classmethod
    def from_data(cls, attributes: dict[str, Any], data=None) -> State:
        """Create the state from attributes and data.

        Args:
            attributes (dict):
                Attributes of the element. This carries information about parameters and
                possibly additional parts that do not depend on time.
            data (:class:`~numpy.ndarray`):
                The numerical data associated with the state of the element
        """
        # re-create the State object using the DictState methods
        obj = super().from_data(attributes, data)
        # set the parameters correctly
        Parameterized.__init__(obj, attributes.get("parameters"))
        return obj  # type: ignore

    @property
    def _data_numba(self) -> tuple:
        """Returns the data associated with the state in a form that numba can
        handle."""
        return tuple(state._data_numba for state in self.data.values())

    @_data_numba.setter
    def _data_numba(self, state_data: tuple) -> None:
        """Sets the data of all states."""
        for key, new_el_data in zip(self.data.keys(), state_data, strict=False):
            self.data[key]._data_numba[...] = new_el_data

    def copy(self, method: Literal["clean", "shallow", "data"] = "clean", data=None):
        """Create a copy of the state.

        Args:
            method (str):
                Determines whether a `clean`, `shallow`, or `data` copy is performed.
                See :meth:`~modelrunner.state.base.StateBase.copy` for details.
            data:
                Data to be used instead of the one in the current state. This data is
                used as is and not copied!

        Returns:
            A copy of the current state object
        """
        # the sole purpose of this method is to set a default copy method
        return super().copy(method, data=data)

    @property
    def elements(self) -> dict[str, _ElementBase]:
        return self.data

    def add_element(self, name: str, element: _ElementBase):
        """Adds an element to the simulation.

        Args:
            name (str):
                The identifier for the element.
            element (:class:`~sim.elements.base._ElementBase`):
                The instance defining the element.
        """
        if name in self.elements:
            self._logger.warning("Overwriting element `%s` in state", name)

        # check dimensionality
        if element.dim is not None:
            if self.dim is None:
                self.dim = element.dim
            elif self.dim != element.dim:
                if self._check_dimension:
                    raise DimensionError(
                        f"Element dimension ({element.dim}) differs from state "
                        f"({self.dim})"
                    )
                else:
                    # report the maximal dimension
                    self.dim = max(self.dim, element.dim)

        self.elements[name] = element

    def get_index(self, name: str) -> int:
        """Returns the numerical index of a specific element.

        Args:
            name (str): The name of the element
        """
        for i, element_name in enumerate(self.elements):
            if name == element_name:
                return i
        raise KeyError(f"`{name}` not in {self.__class__.__name__}")

    def __getitem__(self, key: int | str | Sequence[str]):
        """Extract element by numerical index or by name."""
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

    @property
    def grid(self) -> GridBase:
        """:class:`~pde.grids.base.GridBase`: a grid representing the entire state."""
        grid = None
        for element in self.elements.values():
            # try to find a suitable grid
            try:
                candidate = element.grid  # type: ignore
            except AttributeError:
                pass
            else:
                if (
                    isinstance(candidate, GridBase)
                    and candidate.dim == self.dim
                    and (grid is None or candidate.volume > grid.volume)
                ):
                    grid = candidate

        if grid is None:
            raise RuntimeError("Could not determine suitable grid")
        return grid

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom of the simulation"""
        return sum(element.degrees_of_freedom for element in self.elements.values())

    def get_quantities(self, property_name: str) -> dict[str, Any]:
        """Returns quantities obtained from the elements.

        Quantities are typically implemented as properties or attributes of the
        elements. If an element does not have a property, it is silently ignored and not
        included in the result.

        Args:
            property_name (str):
                The name of the property or attribute that is analyzed

        Returns:
            dict: The value of the quantity is returned for each element. Elements that
            do not define the quantity are not included.
        """
        return {
            element_name: getattr(element, property_name)
            for element_name, element in self
            if hasattr(element, property_name)
        }

    def get_total_quantity(self, property_name: str) -> float:
        """Returns quantities summed over all elements.

        Quantities are typically implemented as properties or attributes. If
        an element does not have a property, it is silently ignored and not
        included in the result.

        Args:
            property_name (str):
                The name of the property or attribute that is analyzed

        Returns:
            float or dict: A total value is returned if total is `True`. Otherwise, the
            value for each element is returned in a dictionary. Note that elements that
            do not define the quantity are not included.
        """
        return sum(self.get_quantities(property_name).values())  # type: ignore

    def get_quantity(self, property_name: str, total: bool = True):
        """Returns quantities obtained from the elements.

        Quantities are typically implemented as properties or attributes. If
        an element does not have a property, it is silently ignored and not
        included in the result.

        Args:
            property_name (str):
                The name of the property or attribute that is analyzed
            total (bool):
                Flag determining whether the sum of all values is returned. If `False`,
                the properties are returned for each element individually.

        Returns:
            float or dict: A total value is returned if total is `True`. Otherwise, the
            value for each element is returned in a dictionary. Note that elements that
            do not define the quantity are not included.

        This function has been deprecated on 2022-06-16
        """
        warnings.warn("method `get_quantity` is deprecated", DeprecationWarning)

        if total:
            return self.get_total_quantity(property_name)
        else:
            return self.get_quantities(property_name)

    def _make_error_estimator(self, backend: str) -> Callable[[Any, Any], float]:
        """Return function that estimates the error between state data.

        Args:
            backend (str): The backend used to calculate the error
        """
        if backend == "numpy":
            el_errs = [
                el._make_error_estimator(backend="numpy") for el in self.values()
            ]

            def state_error_estimator(state1: State, state2: State) -> float:
                """Estimate error for all elements."""
                error = 0.0
                for el_err, el1, el2 in zip(el_errs, state1, state2, strict=False):
                    # The extra cast to a float can be sometimes necessary. We had one
                    # case where the error estimator of an ArrayState returned an record
                    # array, which caused problems downstream. To catch such errors
                    # early, we make an explicit cast here.
                    e = float(el_err(el1, el2))
                    if np.isnan(e):
                        return e
                    else:
                        error = max(error, e)
                return error

            return state_error_estimator

        elif backend == "numba":

            def chain(
                element_id: int, inner: Callable[[Any, Any], float] | None = None
            ) -> Callable[[Any, Any], float]:
                """Recursive factory function for running all actors."""
                # get the evolver function
                el_err = self[element_id]._make_error_estimator(backend="numba")

                @jit
                def wrap(state1, state2) -> float:
                    # get error from inner functions
                    error = 0 if inner is None else inner(state1, state2)
                    # add estiamted error of current element
                    if np.isnan(error):
                        return error
                    else:
                        e = el_err(state1[element_id], state2[element_id])
                        return max(error, e)  # type: ignore

                if element_id < len(self) - 1:
                    # there are more items in the chain
                    return chain(element_id + 1, inner=wrap)
                else:
                    # this is the outermost function
                    return wrap  # type: ignore

            return jit(chain(0))  # type: ignore

        else:
            raise NotImplementedError(f"Unknown backend `{backend}`")

    @plot_on_axes()
    def plot(
        self,
        ax,
        element_args: dict[str, Any] | None = None,
        invisible_elements: Iterable[str] | None = None,
        **kwargs,
    ):
        r"""Visualize the state.

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
            ignore_el: set[str] = self.parameters["invisible_elements"]
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

    def _get_napari_data(self, **kwargs) -> dict[str, dict[str, Any]]:
        r"""Returns data for plotting this state in napari.

        Args:
            \**kwargs: all arguments are forwarded to `_get_napari_layer_data`

        Returns:
            dict: all the information necessary to plot this field
        """
        layers_data = {}
        for name, element in self.elements.items():
            try:
                layer_data = element._get_napari_layer_data(**kwargs)
            except NotImplementedError:
                self._logger.warning(
                    "Element %s does not support interactive plotting", name
                )
            else:
                layers_data[name] = layer_data
        return layers_data

    def plot_interactive(
        self,
        *,
        grid: GridBase | None = None,
        viewer_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Create an interactive plot of the field using :mod:`napari`

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

        # check whether we have enough information to proceed
        if grid is None:
            grid = self.grid
        if grid.dim != self.dim:
            raise RuntimeError(
                "Grid dimension is not compatible (%d != %d)", grid.dim, self.dim
            )

        # try finding the best field that could serve to define the space
        layers_data = self._get_napari_data()

        # do the actual plotting
        with napari_viewer(grid, **viewer_args) as viewer:
            napari_add_layers(viewer, layers_data)
