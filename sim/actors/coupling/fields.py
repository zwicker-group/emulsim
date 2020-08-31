"""
Provides an actor coupling two or more fields

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections import OrderedDict
from typing import Any, Callable, Dict

from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import FieldElementBase
from ..base import ActorBase, ElementsType


class FieldCouplingActor(ActorBase):
    """ actor that couples multiple fields by local interactions """

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

        rhs_expressions: Dict[int, ScalarExpression] = OrderedDict()
        field_names = self.parameters["fields"]
        signature = field_names + ["t"]
        for field_name, rhs in self.parameters["evolution_rates"].items():
            if field_name not in field_names:
                raise RuntimeError(f"Field {field_name} is not in {field_names}")

            field_id = signature.index(field_name)
            rhs_expressions[field_id] = ScalarExpression(rhs, signature)
        self._cache["rhs_expressions"] = rhs_expressions

    def make_evolver_numba(self, fields: ElementsType) -> Callable:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            fields (tuple of :class:`~sim.elements.fields.FieldElementBase`):
                The state of the individual fields

        Returns:
            callable: A function with signature
                (droplets_data: :class:`numpy.ndarray`, field_data, t: float,
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
            """ no-op function serving as innermost nested function """
            pass

        def chain(expression_id, inner) -> Callable:
            """ recursive helper function for running all actors """
            # run through all expressions
            field_id = expressions[expression_id]["field_id"]
            rhs = expressions[expression_id]["rhs"]

            @jit
            def wrap(state_data, t: float, dt: float):
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
            fields[field_id].data += dt * rhs(*field_data, t)
