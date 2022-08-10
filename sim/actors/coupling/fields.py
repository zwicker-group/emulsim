"""
Provides an actor coupling two or more fields

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import functools
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from pde.grids import CartesianGrid
from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import FieldElementBase, ScalarBoundaryFieldElement, ScalarFieldElement
from ..base import ActorBase, ElementsType


class FieldCouplingActor(ActorBase):
    """actor that couples multiple fields by local interactions"""

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

    def __init__(self, parameters: Dict[str, Any] = None):
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
        """prepare the simulation doing pre-calculations

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The state of the individual fields
        """
        # ensure that all grids are compatible
        grid = fields[0].grid  # type: ignore
        for field in fields[1:]:
            grid.assert_grid_compatible(field.grid)  # type: ignore

        rhs_expressions: Dict[int, ScalarExpression] = dict()
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
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
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
                "rhs": rhs.get_compiled(single_arg=False),
            }
            expressions.append(expression_data)

        @jit
        def innermost(state_data, t, dt):
            """no-op function serving as innermost nested function"""
            pass

        def chain(
            expression_id: int,
            inner: Callable[[Tuple[np.ndarray, ...], float, float], None],
        ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
            """recursive helper function for running all actors"""
            # run through all expressions
            field_id = expressions[expression_id]["field_id"]
            rhs = expressions[expression_id]["rhs"]

            @jit
            def wrap(state_data: Tuple[np.ndarray], t: float, dt: float) -> None:
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
        """evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
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


class FieldBoundaryCouplingActor(ActorBase):
    """actor that couples a field with its boundary by local interactions"""

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

    element_classes = (ScalarFieldElement, ScalarBoundaryFieldElement)

    def _update_cache(self, fields: ElementsType) -> None:
        """prepare the simulation doing pre-calculations

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The state of the individual fields
        """
        bulk, boundary = fields
        assert isinstance(bulk.grid, CartesianGrid)  # type: ignore

        assert bulk.dim == boundary.dim
        axis = boundary.parameters["axis"]
        axis_position = boundary.parameters["axis_position"]

        # check whether the boundary is at the upper part of the boundary
        if np.isclose(bulk.grid.axes_bounds[axis][0], axis_position):  # type: ignore
            upper = False
        elif np.isclose(bulk.grid.axes_bounds[axis][1], axis_position):  # type: ignore
            upper = True
        else:
            raise ValueError(f"Position ({axis_position}) is not close to boundary")

        # determine the cell volumes of both fields
        def get_cell_volume(grid: CartesianGrid) -> float:
            cell_volume = functools.reduce(np.outer, grid.cell_volume_data)
            assert cell_volume.size == 1
            return float(np.squeeze(cell_volume))

        self._cache["bulk_volume"] = get_cell_volume(bulk.grid)  # type: ignore
        self._cache["boundary_area"] = get_cell_volume(boundary.grid)  # type: ignore

        # determine the indices to access the bulk concentration close to boundary
        indicies: List[Union[int, slice]] = []
        for i in range(bulk.dim):  # type: ignore
            if i != axis:
                indicies.append(slice(None, None))  # use the full axis (i.e., use `:`)
            elif upper:
                indicies.append(-1)  # use last item
            else:
                indicies.append(0)  # use first item
        self._cache["bulk_boundary_indices"] = tuple(indicies)

        # prepare exchange flux
        expression = self.parameters["exchange_flux"]
        signature = ["bulk", "boundary", "t"]
        self._cache["exchange_flux"] = ScalarExpression(expression, signature)

    def make_evolver_numba(
        self, fields: ElementsType
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The state of the individual fields

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(fields)

        _, boundary = fields
        bulk_boundary_indices = self._cache["bulk_boundary_indices"]
        exchange_flux = self._cache["exchange_flux"].get_compiled()
        thickness = boundary.parameters["thickness"]
        volume_factor = self._cache["boundary_area"] / self._cache["bulk_volume"]

        @jit
        def evolver(
            elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """evolve the flux between bulk and boundary"""
            bulk_data, boundary_data = elements_data

            # determine flux between boundary and
            c_bulk = bulk_data[bulk_boundary_indices]
            c_boundary = boundary_data
            flux = exchange_flux(c_bulk, c_boundary, t)

            # change boundary data
            boundary_data += dt * flux * volume_factor
            bulk_data[bulk_boundary_indices] -= dt * flux / thickness

        # compile the recursive chain
        return evolver  # type: ignore

    def evolve(self, fields: ElementsType, t: float, dt: float) -> None:
        """evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The state of the individual fields
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(fields)
        bulk, boundary = fields

        # determine flux between boundary and
        bulk_boundary_indices = self._cache["bulk_boundary_indices"]
        c_bulk = bulk.data[bulk_boundary_indices]
        c_boundary = boundary.data
        flux = self._cache["exchange_flux"](c_bulk, c_boundary, t)

        # change boundary data
        thickness = boundary.parameters["thickness"]
        volume_factor = self._cache["boundary_area"] / self._cache["bulk_volume"]
        boundary.data[...] += dt * flux * volume_factor
        bulk.data[bulk_boundary_indices] -= dt * flux / thickness


class DomainStitchingActor(ActorBase):
    """actor that couples two domains at a common boundary"""
