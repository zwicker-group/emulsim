"""Provides an actor coupling two or more fields.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable
from types import EllipsisType
from typing import Any

import numpy as np

from pde.backends.numba.utils import jit
from pde.grids import CartesianGrid
from pde.grids.base import DimensionError
from pde.tools.expressions import ScalarExpression

from ... import Parameter
from ...elements import FieldElementBase, MeanfieldElement, ScalarBoundaryFieldElement
from ..base import ActorBase, ElementsSpec, ElementsType


class FieldCouplingActor(ActorBase):
    """Actor coupling multiple fields by local interactions."""

    parameters_default = [
        Parameter(
            "fields",
            ["a", "b"],
            list,
            "The name of the fields that this actor affects.",
        ),
        Parameter(
            "evolution_rates",
            {},
            dict,
            "The expressions determining the dynamics of the fields",
        ),
    ]

    element_classes: tuple[ElementsSpec, ...] | EllipsisType = Ellipsis

    def __init__(self, parameters: dict[str, Any] | None = None):
        """
        Args:
            parameters (dict):
                Parameters defining the behavior of the actor. Call
                :meth:`~ActorBase.show_parameters` for details.
        """
        super().__init__(parameters)

        # check parameter validity
        if len(self.parameters["fields"]) == 0:
            raise ValueError("At least a single field must be given")
        if "t" in self.parameters["fields"]:
            raise ValueError('Field name must not be "t", since this signifies time')

        self.num_fields = len(self.parameters["fields"])
        self.element_classes = (FieldElementBase,) * self.num_fields

    def _update_cache(self, fields: ElementsType) -> None:
        """Prepare the simulation doing pre-calculations.

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
        """
        # ensure that all grids are compatible
        grid = fields[0].grid  # type: ignore
        for field in fields[1:]:
            grid.assert_grid_compatible(field.grid)  # type: ignore

        rhs_expressions: dict[int, ScalarExpression] = {}
        field_names = self.parameters["fields"]
        signature = field_names + ["t"]
        for field_name, rhs in self.parameters["evolution_rates"].items():
            if field_name not in field_names:
                raise RuntimeError(f"Field {field_name} is not in {field_names}")

            field_id = signature.index(field_name)
            rhs_expressions[field_id] = ScalarExpression(rhs, signature)
        self._cache["rhs_expressions"] = rhs_expressions

    def make_evolver_numba(
        self, fields: ElementsType
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """Return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(fields)

        expressions = []
        for field_id, rhs in self._cache["rhs_expressions"].items():
            expression_data = {
                "field_id": field_id,
                "rhs": rhs.get_function(backend="numba", single_arg=False),
            }
            expressions.append(expression_data)

        @jit
        def innermost(state_data, t, dt):
            """No-op function serving as innermost nested function."""

        def chain(
            expression_id: int,
            inner: Callable[[tuple[np.ndarray, ...], float, float], None],
        ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
            """Recursive helper function for running all actors."""
            # run through all expressions
            field_id = expressions[expression_id]["field_id"]
            rhs = expressions[expression_id]["rhs"]

            @jit
            def wrap(state_data: tuple[np.ndarray], t: float, dt: float) -> None:
                inner(state_data, t, dt)
                field_data = state_data[field_id]
                field_data += dt * rhs(*state_data, t)

            if expression_id < len(expressions) - 1:
                # there are more items in the chain
                return chain(expression_id + 1, inner=wrap)
            else:
                # this is the outermost function
                return wrap  # type: ignore

        # compile the recursive chain
        return chain(0, innermost)

    def evolve(self, fields: ElementsType, t: float, dt: float) -> None:
        """Evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(fields)

        # extract the data from all the fields
        field_data = tuple(field.data for field in fields)

        for field_id, rhs in self._cache["rhs_expressions"].items():
            fields[field_id].data[...] += dt * rhs(*field_data, t)


class FieldExchangeActor(ActorBase):
    """Actor exchanging material between two fields on the same grid."""

    parameters_default = [
        Parameter(
            "exchange_rate",
            "0",
            str,
            "The expressions determining the exchange from the first field toward the "
            "second field. The names of the field are set by `field_names`",
        ),
        Parameter(
            "field_name",
            ("c1", "c2"),
            tuple,
            "The names of the two fields, which appear in `exchange_rate`",
        ),
    ]

    element_classes = (FieldElementBase, FieldElementBase)

    def __init__(self, parameters: dict[str, Any] | None = None):
        """
        Args:
            parameters (dict):
                Parameters defining the behavior of the actor. Call
                :meth:`~ActorBase.show_parameters` for details.
        """
        super().__init__(parameters)

        # check parameter validity
        if len(self.parameters["field_name"]) != 2:
            raise ValueError("Exactly two field names expected")
        if "t" in self.parameters["field_name"]:
            raise ValueError('Field name must not be "t", since this signifies time')

    def _update_cache(self, fields: tuple[FieldElementBase, FieldElementBase]) -> None:
        """Prepare the simulation doing pre-calculations.

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
        """
        # ensure that all grids are compatible
        self._cache["grid"] = None
        mean_field = []
        for field in fields:
            if isinstance(field, MeanfieldElement):
                mean_field.append(True)
            else:
                mean_field.append(False)
                grid = field.grid
                if grid is not None:
                    if self._cache["grid"] is not None:
                        grid.assert_grid_compatible(self._cache["grid"])
                    self._cache["grid"] = grid
        self._cache["mean_field"] = tuple(mean_field)
        if all(mean_field):
            # This does not work since they could have different volume, in which case
            # the exchange flux would not be properly defined.
            raise RuntimeError("Cannot exchange flux between two MeanfieldElements")

        # prepare exchange expression
        self._cache["rhs_expression"] = ScalarExpression(
            self.parameters["exchange_rate"],
            signature=tuple(self.parameters["field_name"]) + ("t",),
        )

    def make_evolver_numba(
        self,
        fields: tuple[FieldElementBase, FieldElementBase],  # type: ignore
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """Return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        from pde.backends.numba import numba_backend

        self._check_cache(fields)

        field1_mean, field2_mean = self._cache["mean_field"]
        rhs = self._cache["rhs_expression"].get_function(
            backend="numba", single_arg=False
        )
        integrate = numba_backend.make_integrator(self._cache["grid"])
        if field1_mean:
            add_amount1 = fields[0].make_add_amount_compiled()
        if field2_mean:
            add_amount2 = fields[1].make_add_amount_compiled()

        @jit
        def evolver(
            elements_data: tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """Evolve the flux between bulk and boundary."""
            field1, field2 = elements_data
            flux = rhs(field1, field2, t)

            if field1_mean and field2_mean:
                raise RuntimeError
            elif field1_mean and not field2_mean:
                add_amount1(field1, None, -dt * integrate(flux))
                field2 += dt * flux
            elif not field1_mean and field2_mean:
                field1 -= dt * flux
                add_amount2(field2, None, dt * integrate(flux))
            else:
                field1 -= dt * flux
                field2 += dt * flux

        return evolver  # type: ignore

    def evolve(
        self,
        fields: tuple[FieldElementBase, FieldElementBase],  # type: ignore
        t: float,
        dt: float,
    ) -> None:
        """Evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(fields)

        # extract the data from all the fields
        field1 = fields[0].data
        field2 = fields[1].data
        field1_mean, field2_mean = self._cache["mean_field"]

        # calculate the exchange rate
        flux = self._cache["rhs_expression"](field1, field2, t)

        # update the fields accordingly
        if field1_mean and field2_mean:
            raise RuntimeError
        elif field1_mean and not field2_mean:
            fields[0].add_amount(None, -dt * self._cache["grid"].integrate(flux))  # type: ignore
            field2 += dt * flux
        elif not field1_mean and field2_mean:
            field1 -= dt * flux
            fields[1].add_amount(None, dt * self._cache["grid"].integrate(flux))  # type: ignore
        else:
            field1 -= dt * flux
            field2 += dt * flux


class FieldBoundaryExchangeActor(ActorBase):
    """Actor exchanging material between a field and its boundary.

    This actor does move material between support points in the boundary field and the
    adjacent support points in the bulk field. This is an approximation, which might
    lead to unphysical situations since material is injected half a discretization size
    away from the boundary (at the first support point) instead of directly at the
    boundary via a flux boundary conditions. However, the advantage of this method is
    that it is suitable for arbitrary PDEs describing the bulk and always ensures
    material conservation.

    Note:
        The flux is oriented such that positive values move material from the bulk to
        the boundary. The expression of the exchange flux may depend on the
        concentrations in the bulk and the boundary, which are available as the
        variables :code:`bulk` and :code:`boundary` in the respective expression
        parameter. In contrast, the names of the actual elements in the entire
        simulation (e.g., `cytosol` and `membrane`) cannot be used to refer to these
        concentrations. The flux is an area flux, so that the total amount of material
        transferred between the two fields is proportional to the time step and the
        boundary area.
    """

    parameters_default = [
        Parameter(
            "exchange_flux",
            "0",
            str,
            "The expressions determining the flux from the bulk to the boundary. The "
            "expression may depend on the concentration in the bulk (`bulk`), the "
            "concentration in the boundary (`boundary`), and explicit time (`t`).",
        ),
    ]

    element_classes = (FieldElementBase, ScalarBoundaryFieldElement)

    def _update_cache(self, fields: ElementsType) -> None:
        """Prepare the simulation doing pre-calculations.

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
        """
        bulk, boundary = fields
        if bulk.dim != boundary.dim:
            raise DimensionError(f"Bulk != boundary ({bulk.dim} != {boundary.dim})")

        bulk_grid, boundary_grid = bulk.grid, boundary.grid  # type: ignore
        if not isinstance(bulk_grid, CartesianGrid):
            raise TypeError("Bulk must be defined on CartesianGrid")
        axis = boundary.axis  # type: ignore
        axis_position = boundary.parameters["axis_position"]

        # check whether the boundary is at the upper part of the boundary
        if np.isclose(bulk_grid.axes_bounds[axis][0], axis_position):
            upper = False
        elif np.isclose(bulk_grid.axes_bounds[axis][1], axis_position):
            upper = True
        else:
            raise ValueError(f"Position ({axis_position}) is not close to boundary")

        # determine the cell volumes and areas of both fields
        def get_cell_volume(grid: CartesianGrid) -> float:
            assert np.isscalar(grid.cell_volume_data[0])
            return np.prod(grid.cell_volume_data)  # type: ignore

        self._cache["bulk_cell_volume"] = get_cell_volume(bulk_grid)
        self._cache["boundary_cell_area"] = get_cell_volume(boundary_grid)
        self._cache["boundary_cell_volume"] = (
            self._cache["boundary_cell_area"] * boundary.parameters["thickness"]
        )

        # determine whether the grids are directly compatible
        indices = tuple(i for i in range(bulk_grid.num_axes) if i != axis)
        try:
            sub_grid = bulk_grid.slice(indices)
        except AttributeError:
            # fall-back for deprecated method (remove on 2023-03-15)
            sub_grid = bulk_grid.get_subgrid(indices)  # type: ignore
        if not np.allclose(boundary_grid.axes_bounds, sub_grid.axes_bounds):
            self._logger.warning("Field extents are incompatible")

        if boundary_grid.compatible_with(sub_grid):
            # the grids are compatible
            self._cache["grid_match"] = "exact"

            # determine the indices to access the bulk concentration close to boundary
            indicies: list[int | slice] = []
            for i in range(bulk.dim):  # type: ignore
                if i != axis:
                    indicies.append(slice(None, None))  # use the full axis (i.e., `:`)
                elif upper:
                    indicies.append(-1)  # use last item
                else:
                    indicies.append(0)  # use first item
            self._cache["bulk_boundary_indices"] = tuple(indicies)

        elif boundary_grid.shape >= sub_grid.shape:
            # the boundary grid is resolved more finely
            self._cache["grid_match"] = "boundary_resolved"
            self._cache["bulk_coordinates"] = boundary.bulk_coordinates  # type: ignore

        elif boundary_grid.shape <= sub_grid.shape:
            # the bulk grid is resolved more finely
            self._cache["grid_match"] = "bulk_resolved"

        else:
            # a mixed situation with different resolutions
            self._cache["grid_match"] = "mixed"

        # prepare exchange flux
        expression = self.parameters["exchange_flux"]
        signature = ["bulk", "boundary", "t"]
        self._cache["exchange_flux"] = ScalarExpression(expression, signature)

    def make_evolver_numba(
        self, fields: ElementsType
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """Return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(fields)

        bulk, boundary = fields
        exchange_flux = self._cache["exchange_flux"].get_function(backend="numba")
        bulk_cell_volume = self._cache["bulk_cell_volume"]
        boundary_cell_area = self._cache["boundary_cell_area"]
        boundary_cell_volume = self._cache["boundary_cell_volume"]

        if self._cache["grid_match"] == "exact":
            # exchange material between corresponding points

            bulk_boundary_indices = self._cache["bulk_boundary_indices"]

            @jit
            def evolver(
                elements_data: tuple[np.ndarray, np.ndarray], t: float, dt: float
            ) -> None:
                """Evolve the flux between bulk and boundary."""
                bulk_data, boundary_data = elements_data

                # determine concentrations in both fields
                c_bulk = bulk_data[bulk_boundary_indices]
                c_boundary = boundary_data

                # determine flux between boundary and bulk
                flux = exchange_flux(c_bulk, c_boundary, t)
                exchange_amount = flux * dt * boundary_cell_area

                # exchange this amount between the fields
                bulk_data[bulk_boundary_indices] -= exchange_amount / bulk_cell_volume
                boundary_data += exchange_amount / boundary_cell_volume

        elif self._cache["grid_match"] == "boundary_resolved":
            # use interpolation in the under-resolved bulk

            bulk_coordinates = self._cache["bulk_coordinates"]
            bulk_interpolator = bulk.make_get_concentration_compiled()  # type: ignore
            bulk_add_amount = bulk.make_add_amount_compiled()  # type: ignore
            bndry_shape = boundary.grid.shape  # type: ignore

            @jit
            def evolver(
                elements_data: tuple[np.ndarray, np.ndarray], t: float, dt: float
            ) -> None:
                """Evolve the flux between bulk and boundary."""
                bulk_data, boundary_data = elements_data

                # determine concentrations at all bulk points. This cannot be done in
                # the next loop since bulk concentrations are modified by the loop.
                c_bulk = np.empty(bndry_shape)
                for bndry_idx in np.ndindex(*bndry_shape):
                    bulk_point = bulk_coordinates[bndry_idx]
                    c_bulk[bndry_idx] = bulk_interpolator(bulk_data, bulk_point)

                # iterate over all boundary points
                for bndry_idx in np.ndindex(*bndry_shape):
                    bulk_point = bulk_coordinates[bndry_idx]
                    c_boundary = boundary_data[bndry_idx]

                    # determine flux between boundary and bulk
                    flux = exchange_flux(c_bulk[bndry_idx], c_boundary, t)
                    exchange_amount = flux * dt * boundary_cell_area

                    # exchange this amount between the fields
                    bulk_add_amount(bulk_data, bulk_point, -exchange_amount)
                    boundary_data[bndry_idx] += exchange_amount / boundary_cell_volume

        else:
            raise NotImplementedError

        return evolver  # type: ignore

    def evolve(self, fields: ElementsType, t: float, dt: float) -> None:
        """Evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~emulsim.elements.fields.FieldElementBase`):
                The state of the individual fields
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(fields)
        bulk, boundary = fields

        # determine concentrations in both fields
        if self._cache["grid_match"] == "exact":
            bulk_boundary_indices = self._cache["bulk_boundary_indices"]
            c_bulk = bulk.data[bulk_boundary_indices]
        elif self._cache["grid_match"] == "boundary_resolved":
            c_bulk = bulk.get_concentration(self._cache["bulk_coordinates"])  # type: ignore
        else:
            raise NotImplementedError
        c_boundary = boundary.data

        # determine flux between boundary and bulk
        flux = self._cache["exchange_flux"](c_bulk, c_boundary, t)
        exchange_amount = flux * dt * self._cache["boundary_cell_area"]

        # exchange this amount between the fields
        if self._cache["grid_match"] == "exact":
            # exchange material point-wise
            bulk_conc_change = exchange_amount / self._cache["bulk_cell_volume"]
            bulk.data[bulk_boundary_indices] -= bulk_conc_change

        elif self._cache["grid_match"] == "boundary_resolved":
            # insert the correct amount at each boundary position
            bulk_coordinates = self._cache["bulk_coordinates"]
            for index, amount in np.ndenumerate(exchange_amount):
                bulk.add_amount(bulk_coordinates[index], -amount)  # type: ignore
        boundary.data[...] += exchange_amount / self._cache["boundary_cell_volume"]
