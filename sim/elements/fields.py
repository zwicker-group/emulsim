"""
Provides elements that represent extended, discretized fields 

.. autosummary::
   :nosignatures:

   ~MeanfieldElement
   ~ReservoirElement
   ~ScalarFieldElement
   ~FieldCollectionElement

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, Sequence, Tuple

import numba as nb
import numpy as np
from numba.extending import register_jitable

from pde.fields import FieldCollection, ScalarField
from pde.grids.cartesian import CartesianGrid, GridBase
from pde.tools.cache import cached_property
from pde.tools.cuboid import Cuboid
from pde.tools.numba import jit
from pde.tools.parameters import Parameter
from pde.tools.plotting import plot_on_axes
from pde.tools.typing import NumberOrArray

from .base import ElementBase


class ReservoirElement(ElementBase):
    """an element representing a homogeneous, constant field"""

    dim = None  # works for any dimension

    def __init__(self, data: float = 0, parameters: Dict[str, Any] = None):
        """
        Args:
            data (float):
                The concentration in the field
        """
        super().__init__(np.full((1,), data, dtype=np.double), parameters)

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        return 0

    @property
    def concentration(self) -> float:
        """float: the concentration in the field"""
        return float(self.data[0])

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.concentration})"

    def __str__(self):
        return f"{self.__class__.__name__}(data={self.concentration})"

    @plot_on_axes()
    def plot(self, ax, color="tab:blue", **kwargs):
        """plot the field

        Args:
            color:
                The color in which the field is shown. All matplotlib
                color specifications are allowed.
            {PLOT_ARGS}
        """
        pass

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
        pass

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

        @nb.jit
        def add_amount(data: np.ndarray, point: np.ndarray, amount: float):
            pass

        return add_amount  # type: ignore


class FieldElementBase(ElementBase, metaclass=ABCMeta):
    """base class for field elements"""

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

    @cached_property()
    def grid(self) -> CartesianGrid:
        """:class:`pde.grids.cartesian.CartesianGrid`: discretization grid"""
        return CartesianGrid(self.bounds, 1)

    @abstractproperty
    def total_amount(self) -> float:
        """float: the total material amount in the field"""
        pass

    @property
    def average_concentration(self) -> float:
        """float: the average material concentration in the field"""
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
    """an element representing a homogeneous, changing field"""

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

    @classmethod
    def from_field(cls, field: ScalarField) -> MeanfieldElement:
        """create a mean field element from a scalar field

        Args:
            field (:class:`~pde.fields.scalar.ScalarField`):
                The scalar field that initializes the element

        Returns:
            :class:`MeanfieldElement`: The initialized instance
        """
        assert isinstance(field, ScalarField)
        data = float(field.average)  # type: ignore
        return cls(data, {"bounds": field.grid.axes_bounds})

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        return 1

    @property
    def concentration(self) -> float:
        """float: the concentration in the field"""
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
        """:class:`~pde.fields.scalar.ScalarField`: representation as a scalar field"""
        return ScalarField(self.grid, data=self.concentration)

    @property
    def total_amount(self) -> float:
        """float: the total material amount in the field"""
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
    """the state associated with a spatially resolved field"""

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
        super().__init__(None, parameters)

        if not isinstance(self.grid, CartesianGrid):
            raise NotImplementedError(
                "The simulations are only been implemented for Cartesian grids and not "
                f"for {self.grid.__class__.__name__}"
            )

        self._field = ScalarField(self.grid, data, label=self.parameters["label"])
        self._data = self._field.data
        self.set_bounds(self.grid.axes_bounds)

    @classmethod
    def from_field(cls, field: ScalarField) -> ScalarFieldElement:
        """create a scalar field element from a scalar field

        Args:
            field (:class:`~pde.fields.scalar.ScalarField`):
                The scalar field that initializes the element

        Returns:
            :class:`ScalarFieldElement`: The initialized instance
        """
        assert isinstance(field, ScalarField)
        return cls(field.data, {"grid": field.grid, "label": field.label})

    @property
    def grid(self) -> CartesianGrid:
        """:class:`~pde.grids.cartesian.CartesianGrid`: discretization grid"""
        return self.parameters["grid"]  # type: ignore

    @property
    def field(self) -> ScalarField:
        """:class:`~pde.fields.scalar.ScalarField`: the scalar field"""
        return self._field

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        return self.data.size

    def plot(self, ax=None, **kwargs):
        """plot the field

        This simply calls :meth:`~pde.fields.base.DataFieldBase.plot` and all
        arguments are forwarded to this method.
        """
        return self._field.plot(ax=ax, **kwargs)

    @property
    def total_amount(self) -> float:
        """float: the total material amount in the field"""
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
        interpolate = self._field.make_interpolator(backend="numba")

        @register_jitable
        def get_concentration(data: np.ndarray, point: np.ndarray):
            """helper function swapping the argument order"""
            return interpolate(point, data)

        return get_concentration  # type: ignore

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amount: float), which adds `amount`
            to the field state given by `data` at point `point`.
        """
        return self._field.grid.make_inserter_compiled()


class FieldCollectionElement(ElementBase):
    """the state associated with multiple spatially resolved fields"""

    parameters_default = [
        Parameter(
            "grid",
            None,
            description="The grid on which the fields are discretized. The grid also "
            "determines the space dimension and its extension.",
            extra={
                "serializer": lambda grid: grid.state_serialized,
                "unserializer": GridBase.from_state,
            },
        ),
        Parameter("label", "", str, "The name of the field collection"),
    ]

    def __init__(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any] = None,
    ):
        """
        Args:
            num_fields (int):
                The number of fields in the collection
            data (:class:`~numpy.ndarray`):
                Field values at the support points of the grid
            parameters (dict):
                Additional parameters determining how the element behaves. Most
                importantly, the entry 'grid' determines the discretization grid
                on which this field is defined.
        """
        super().__init__(None, parameters)

        if not isinstance(self.grid, CartesianGrid):
            raise NotImplementedError(
                "The simulations are only been implemented for Cartesian grids and not "
                f"for {self.grid.__class__.__name__}"
            )

        data = np.asarray(data)
        if data.size == 1:
            fields = [ScalarField(self.grid, data)]
        else:
            fields = [ScalarField(self.grid, field_data) for field_data in data]
        self._field = FieldCollection(fields, label=self.parameters["label"])

        self._data = self._field.data

        self._cuboid = Cuboid.from_bounds(
            np.array(self.grid.axes_bounds, np.double), mutable=False
        )
        self.dim: int = self._cuboid.dim
        self.bounds = self._cuboid.bounds
        self.volume = float(self._cuboid.volume)

    @property
    def num_fields(self) -> int:
        """int: the number of fields described by this collection"""
        return len(self._field)

    @classmethod
    def from_field(cls, field: FieldCollection) -> FieldCollectionElement:
        """create a scalar field element from a scalar field

        Args:
            field (:class:`~pde.fields.collection.FieldCollection`):
                The field collection that initializes the element

        Returns:
            :class:`FieldCollectionElement`: The initialized instance
        """
        for f in field:
            assert isinstance(f, ScalarField)
        return cls(
            data=field.data,
            parameters={"grid": field.grid, "label": field.label},
        )

    @property
    def grid(self) -> CartesianGrid:
        """:class:`~pde.grids.cartesian.CartesianGrid`: discretization grid"""
        return self.parameters["grid"]  # type: ignore

    @property
    def field(self) -> FieldCollection:
        """:class:`~pde.fields.scalar.ScalarField`: the scalar field"""
        return self._field

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        return self.data.size

    def plot(self, ax=None, **kwargs):
        """plot the field

        Note that only the first field is plotted if the dimension is different from 1.
        The method simply calls :meth:`~pde.fields.base.DataFieldBase.plot` and all
        arguments are forwarded.
        """
        if self.dim == 1:
            for field in self._field:
                field.plot(ax=ax, **kwargs)
        else:
            self._field[0].plot(ax=ax, **kwargs)

    @property
    def amounts(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the total material amount in each field"""
        return np.array(self._field.integrals)

    @property
    def total_amount(self) -> float:
        """float: the total material amount in the field"""
        return np.sum(self._field.integrals).real  # type: ignore

    def get_concentrations(self, points: np.ndarray):
        """determine concentrations at the given points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentrations are returned
        """
        return np.array([field.interpolate(points) for field in self._field])

    def add_amounts(self, point: np.ndarray, amounts: np.ndarray):
        """add the given amounts to the fields

        Args:
            point (:class:`~numpy.ndarray`):
                Point where the amounts are added to the fields
            amounts (:class:`~numpy.ndarray`):
                The total amount added to each field
        """
        for field, amount in zip(self._field, amounts):
            field.insert(point, amount)

    def make_get_concentrations_compiled(self) -> Callable:
        """get a compiled function for obtaining concentrations

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`), which determines the concentrations
            at point `point` given the field state `data`.
        """
        # we just need one interpolator for all fields since they are assumed to be
        # equivalent, e.g., lie on the same grid (and have the same rank)
        interpolate = self._field[0].make_interpolator(backend="numba")
        num_fields = self.num_fields

        @register_jitable
        def get_concentration(data: np.ndarray, point: np.ndarray) -> np.ndarray:
            """helper function swapping the argument order"""
            result = np.empty(num_fields)
            for i in range(num_fields):
                result[i] = interpolate(point, data[i])
            return result

        return get_concentration  # type: ignore

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amounts: :class:`~numpy.ndarray`), which
            adds `amounts` to the field state given by `data` at point `point`.
        """
        # we just need one inserter for all fields since they are assumed to be
        # equivalent, e.g., lie on the same grid (and have the same rank)
        inserter_single = self._field[0].grid.make_inserter_compiled()
        num_fields = self.num_fields

        @register_jitable
        def inserter(data: np.ndarray, point: np.ndarray, amounts: np.ndarray) -> None:
            """helper function inserting amounts"""
            for i in range(num_fields):
                inserter_single(data[i], point, amounts[i])

        return inserter  # type: ignore


class ScalarBoundaryFieldElement(ScalarFieldElement):
    """the state associated with a spatially resolved boundary

    Note:
        The data described by this element are volume concentrations with units
        `length**-dim`, where `dim` is the dimension of the bulk (so the boundary has
        dimensions `dim - 1`). To convert the concentration in a particular cell into a
        total amount it has to be multiplied by the cell volume and the thickness of the
        boundary.
    """

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
        Parameter(
            "axis", -1, int, description="Axis along which the boundary is placed"
        ),
        Parameter(
            "axis_position",
            math.nan,
            float,
            description="Position of the boundary along the axis. If omitted, the "
            "boundary might not support some operations.",
        ),
        Parameter(
            "thickness",
            1,
            float,
            description="Thickness of the boundary, which affects calculations of "
            "total amounts and potentially boundary conditions",
        ),
        Parameter(
            "plot_thickness",
            1,
            float,
            description="Thickness used when plotting the boundary",
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
        super().__init__(None, parameters)  # type: ignore

        if not 0 <= self.axis <= self.grid.num_axes:
            raise ValueError(f"`axis={self.axis}` is out of bounds")

        if not isinstance(self.grid, CartesianGrid):
            raise NotImplementedError(
                "The simulations are only been implemented for Cartesian grids and not "
                f"for {self.grid.__class__.__name__}"
            )

        self._field = ScalarField(self.grid, data, label=self.parameters["label"])
        self._data = self._field.data
        self.set_bounds(self.grid.axes_bounds)

        # correct some values to make them bulk quantities
        self.dim = self.grid.dim + 1
        self.volume = self._cuboid.volume * self.parameters["thickness"]

    @property
    def axis(self) -> int:
        """int: the axis of the full domain that this boundary is associated with"""
        axis = int(self.parameters["axis"])
        if axis < 0:
            axis += self.grid.dim
        return axis

    @classmethod
    def from_field(cls, field: ScalarField) -> ScalarBoundaryFieldElement:
        """create a scalar boundary element from a scalar field

        Args:
            field (:class:`~pde.fields.scalar.ScalarField`):
                The scalar field that initializes the element

        Returns:
            :class:`ScalarFieldElement`: The initialized instance
        """
        raise NotImplementedError  # overwrite inherited method

    @classmethod
    def from_bulk_grid(
        cls,
        grid: CartesianGrid,
        axis: int,
        upper: bool = None,
        data: NumberOrArray = 0,
        parameters: Dict[str, Any] = None,
    ) -> ScalarBoundaryFieldElement:
        """create a scalar boundary element using a grid describing the full domain

        Args:
            grid (:class:`~pde.grids.CartesianGrid`):
                The scalar field describing the full domain
            axis (int):
                The axis along which the boundary is initialized
            upper (bool):
                Specified whether the upper or lower boundary along the given axis is
                specified by this field.
            data (:class:`~numpy.ndarray` or float, optional):
                Field values at the support points of the grid
            parameters (dict):
                Additional parameters determining how the element behaves.
        """
        if parameters is None:
            parameters = {}
        for key in ["grid", "axis", "axis_position"]:
            if key in parameters:
                raise ValueError(f"`{key}` parameter not accepted by `from_bulk_grid`")

        if axis < 0:
            axis += grid.num_axes
        indices = tuple(i for i in range(grid.num_axes) if i != axis)
        try:
            parameters["grid"] = grid.slice(indices)
        except AttributeError:
            # fall-back for deprecated method (remove on 2023-03-15)
            parameters["grid"] = grid.get_subgrid(indices)  # type: ignore
        parameters["axis"] = axis
        if upper is None:
            parameters["axis_position"] = math.nan
        elif upper is True:
            parameters["axis_position"] = grid.axes_bounds[axis][1]
        elif upper is False:
            parameters["axis_position"] = grid.axes_bounds[axis][0]
        else:
            raise TypeError
        return cls(data, parameters)

    @cached_property()
    def bulk_coordinates(self) -> np.ndarray:
        """:class:`~numpy.ndarray` all boundary points in the bulk coordinate system"""
        axis_position = self.parameters["axis_position"]
        if np.isnan(axis_position):
            raise RuntimeError("Axis position was not specified")
        return np.insert(self.grid.cell_coords, self.axis, axis_position, axis=-1)

    @plot_on_axes()
    def plot(self, ax=None, colorbar: bool = False, **kwargs):
        """plot the boundary field element

        Args:
            {PLOT_ARGS}
            colorbar (bool):
                Flag determining whether a colorbar is shown.
            **kwargs:
                All remaining parameters are forwarded to
                :class:`matplotlib.axes.Axes.pcolormesh`
        """
        if self.dim == 2:
            # plot boundary of 2d domain as a line
            axis_position = self.parameters["axis_position"]
            plot_thickness = self.parameters["plot_thickness"]
            if np.isnan(axis_position):
                self._logger.debug("Cannot plot since `axis_position` not specified")
                return  # silently fail

            # determine coordinates
            x = self.grid.axes_coords[0]
            dx2 = self.grid.discretization[0] / 2
            xs_half = np.r_[x[0] - dx2, x + dx2]
            xs = np.c_[xs_half, xs_half]

            ys = np.c_[
                np.full_like(xs_half, axis_position - plot_thickness / 2),
                np.full_like(xs_half, axis_position + plot_thickness / 2),
            ]

            if self.axis == 0:
                xs, ys = ys.T, xs.T
                data = self._field.data.reshape(1, -1)
            elif self.axis == 1:
                data = self._field.data.reshape(-1, 1)
            else:
                raise RuntimeError("`axis` value out of bounds")

            # show concentration along the line
            colormesh = ax.pcolormesh(xs, ys, data, shading="flat", **kwargs)
            if colorbar:
                from pde.tools.plotting import add_scaled_colorbar

                add_scaled_colorbar(colormesh, ax=ax)

        else:
            raise NotImplementedError(f"Plotting dim={self.dim} not implemented")

    def add_amount(self, point: np.ndarray, amount: float):
        """add the given amount to the field

        Args:
            point (:class:`~numpy.ndarray`):
                Point where the amount is added to the field
            amount (float):
                The total amount added to the field
        """
        self._field.insert(point, amount / self.parameters["thickness"])

    def make_add_amount_compiled(self) -> Callable:
        """get a compiled function for adding amount to the field

        Returns:
            callable: a function with signature (data: :class:`~numpy.ndarray`,
            point: :class:`~numpy.ndarray`, amount: float), which adds `amount`
            to the field state given by `data` at point `point`.
        """
        inserter = self._field.grid.make_inserter_compiled()
        thickness = self.parameters["thickness"]

        @jit
        def insert(data: np.ndarray, point: np.ndarray, amount: NumberOrArray) -> None:
            inserter(data, point, amount / thickness)

        return insert  # type: ignore

    def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
