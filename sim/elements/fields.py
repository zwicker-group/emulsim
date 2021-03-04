"""
Provides elements that represent extended, discretized fields 

.. autosummary::
   :nosignatures:

   ~ScalarFieldElement
   ~MeanfieldElement

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, Sequence, Tuple

import numba as nb
import numpy as np

from pde.fields import ScalarField
from pde.grids import CartesianGrid
from pde.grids.cartesian import CartesianGridBase, GridBase
from pde.tools.cuboid import Cuboid
from pde.tools.parameters import Parameter
from pde.tools.plotting import plot_on_axes
from pde.tools.typing import NumberOrArray

from .base import ElementBase


class FieldElementBase(ElementBase, metaclass=ABCMeta):
    """ base class for field elements """

    def set_bounds(self, bounds: Sequence[Tuple[float, float]]) -> None:
        """set the boundaries of the field

        Args:
            bounds (sequence):
                A sequence of tuples specifying the lower and upper bound for
                each axis. The number of entries sets the space dimension.
        """
        self._cuboid = Cuboid.from_bounds(np.array(bounds, np.double), mutable=False)
        self.dim: int = self._cuboid.dim
        self.bounds = self._cuboid.bounds
        self.volume = float(self._cuboid.volume)

    @property
    def grid(self) -> CartesianGrid:
        """ :class:`pde.grids.cartesian.CartesianGrid`: discretization grid """
        return CartesianGrid(self.bounds, 1)

    @abstractproperty
    def total_amount(self) -> float:
        """ float: the total material amount in the field """
        pass

    @property
    def average_concentration(self) -> float:
        """ float: the average material concentration in the field """
        return self.total_amount / self.volume

    @abstractmethod
    def get_concentration(self, points):
        """determine concentration at the given points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        pass

    @abstractmethod
    def add_amount(self, point: np.ndarray, amount: float):
        """add the given amount to the field

        Args:
            point (:class:`~numpy.ndarray`):
                Point where the amount is added to the field
            amount (float):
                The total amount added to the field
        """
        pass

    def make_get_concentration_compiled(self) -> Callable:
        """get a compiled function for obtaining concentrations

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`), which determines the concentration
            at point `point` given the field state `data`.
        """
        raise NotImplementedError

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amount: float), which adds `amount`
            to the field state given by `data` at point `point`.
        """
        raise NotImplementedError

    def plot(self, ax=None, **kwargs):
        """plot the field"""
        pass

    def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

        Args:
            **kwargs: Extra arguments are passed to plotting function

        Returns:
            dict: all the information necessary to plot this field
        """
        return self.field._get_napari_layer_data(**kwargs)  # type: ignore


class MeanfieldElement(FieldElementBase):
    """ an element representing a homogeneous field """

    parameters_default = [
        Parameter(
            "bounds",
            None,
            description="Sets the size of the Cartesian space covered by this element. "
            "This should be a list of tuples, where each element denotes the lower and "
            "upper bounds of an axis. The number of elements then determines the "
            "dimension of the space",
        )
    ]

    def __init__(self, data: float = 0, parameters: Dict[str, Any] = None):
        """initialize the meanfield element

        Args:
            data (float):
                The initial concentration in the field
            parameters (dict):
                Additional parameters determining how the element behaves. Most
                importantly, the entry 'bounds' determines the size of the
                element. It needs to be a sequence of tuples specifying the
                lower and upper bound for each axis. The number of entries sets
                the space dimension.
        """
        super().__init__(np.full((1,), data, dtype=np.double), parameters)

        # store data in a mutable 1d-array
        if self.parameters["bounds"] is None:
            raise ValueError("`bounds` need to be specified in parameters")
        else:
            self.set_bounds(self.parameters["bounds"])

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom for this element """
        return 1

    @property
    def concentration(self) -> float:
        """ float: the concentration in the field """
        return float(self.data[0])

    @concentration.setter
    def concentration(self, value: float):
        """set the field concentration

        Args:
            value (float):
                The new concentration
        """
        self.data[0] = value

    @property
    def field(self) -> ScalarField:
        """:class:`~pde.fields.scalar.ScalarField`: representation as a scalar field """
        return ScalarField(self.grid, data=self.concentration)

    @property
    def total_amount(self) -> float:
        """ float: the total material amount in the field """
        return self.concentration * self.volume

    @total_amount.setter
    def total_amount(self, amount: float):
        """set the total material amount in the field

        Args:
            amount (float):
                The new total amount
        """
        self.concentration = amount / self.volume

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bounds={self.bounds!r}, "
            f"data={self.concentration})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(bounds={self.bounds!s}, "
            f"data={self.concentration})"
        )

    @plot_on_axes()
    def plot(self, ax, color="tab:blue", **kwargs):
        """plot the field

        Args:
            color:
                The color in which the field is shown. All matplotlib
                color specifications are allowed.
            {PLOT_ARGS}
        """
        # create the rectangle representing the background
        from matplotlib import patches

        rect = patches.Rectangle(
            self._cuboid.pos[:2],
            *self._cuboid.size[:2],
            edgecolor="none",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.set_xlim(*self.bounds[0])
        ax.set_ylim(*self.bounds[1])
        ax.set_aspect(1)

    def get_concentration(self, points: np.ndarray):
        """determine concentration at the given points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        points = np.asanyarray(points)
        if points.ndim == 1:
            # a single point
            return self.concentration
        elif points.ndim == 2:
            # many points
            return np.full(len(points), self.concentration)
        else:
            raise ValueError("Expected single point of list of points")

    def add_amount(self, point: np.ndarray, amount: float):
        """add the given amount to the field

        Args:
            point:
                Not used and only retained to match the interface
            amount:
                The total amount added to the field
        """
        self.data[0] += amount / self.volume

    def make_get_concentration_compiled(self) -> Callable:
        """get a compiled function for obtaining concentrations

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`), which determines the concentration
            at point `point` given the field state `data`.
        """

        @nb.jit
        def get_concentration(data: np.ndarray, point: np.ndarray):
            return data[0]

        return get_concentration  # type: ignore

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amount: float), which adds `amount`
            to the field state given by `data` at point `point`.
        """
        volume = self.volume

        @nb.jit
        def add_amount(data: np.ndarray, point: np.ndarray, amount: float):
            data += amount / volume

        return add_amount  # type: ignore


class ScalarFieldElement(FieldElementBase):
    """ the state associated with a spatially resolved field """

    parameters_default = [
        Parameter(
            "grid",
            None,
            description="The grid on which the field is discretized. The grid also "
            "determines the space dimension and its extension.",
            extra={
                "serializer": lambda grid: grid.state_serialized,
                "unserializer": GridBase.from_state,
            },
        ),
        Parameter("label", "", str, "The name of the field"),
    ]

    def __init__(self, data: NumberOrArray = 0, parameters: Dict[str, Any] = None):
        """
        Args:
            data (:class:`~numpy.ndarray` or float, optional):
                Field values at the support points of the grid
            parameters (dict):
                Additional parameters determining how the element behaves. Most
                importantly, the entry 'grid' determines the discretization grid
                on which this field is defined.
        """
        # set temporary data first and overwrite it later
        super().__init__(np.empty(()), parameters)

        if not isinstance(self.grid, CartesianGridBase):
            raise NotImplementedError(
                "The simulations are only been "
                "implemented for Cartesian grids and not "
                f"for {self.grid.__class__.__name__}"
            )

        self._field = ScalarField(self.grid, data, label=self.parameters["label"])
        self._data = self._field.data
        self.set_bounds(self.grid.axes_bounds)

    @classmethod
    def from_field(cls, field: ScalarField) -> "ScalarFieldElement":
        """create a scalar field element from a scalar field

        Args:
            field (:class:`~pde.fields.scalar.ScalarField`):
                The scalar field that initializes the element

        Returns:
            :class:`ScalarFieldElement`: The initialized instance
        """
        return cls(field.data, {"grid": field.grid, "label": field.label})

    @property
    def grid(self) -> CartesianGrid:
        """ :class:`~pde.grids.cartesian.CartesianGrid`: discretization grid """
        return self.parameters["grid"]  # type: ignore

    @property
    def field(self) -> ScalarField:
        """ :class:`~pde.fields.scalar.ScalarField`: the scalar field """
        return self._field

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom for this element """
        return int(np.product(self.grid.shape))

    def plot(self, ax=None, **kwargs):
        """plot the field

        This simply calls :meth:`~pde.fields.base.DataFieldBase.plot` and all
        arguments are forwarded to this method.
        """
        return self._field.plot(ax=ax, **kwargs)

    @property
    def total_amount(self) -> float:
        """ float: the total material amount in the field """
        return self._field.integral.real

    def get_concentration(self, points: np.ndarray):
        """determine concentration at the given points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        return self._field.interpolate(points)

    def add_amount(self, point: np.ndarray, amount: float):
        """add the given amount to the field

        Args:
            point (:class:`~numpy.ndarray`):
                Point where the amount is added to the field
            amount (float):
                The total amount added to the field
        """
        self._field.insert(point, amount)

    def make_get_concentration_compiled(self) -> Callable:
        """get a compiled function for obtaining concentrations

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`), which determines the concentration
            at point `point` given the field state `data`.
        """
        return self._field.grid.make_interpolator_compiled()

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amount: float), which adds `amount`
            to the field state given by `data` at point `point`.
        """
        return self._field.grid.make_inserter_compiled()
