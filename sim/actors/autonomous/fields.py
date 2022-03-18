"""
Provides actors that influence scalar fields

.. autosummary::
   :nosignatures:

   ~MeanfieldActor
   ~ScalarPDEActor
   ~DiffusionActor
   ~ReactionDiffusionActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Tuple, Type  # @UnusedImport

import numba as nb
import numpy as np

from pde.pdes.base import PDEBase
from pde.tools.docstrings import get_text_block
from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import MeanfieldElement, ScalarFieldElement
from ..base import ActorBase, ElementsType


class MeanfieldActor(ActorBase):
    """actor simulating mean field chemical reactions"""

    parameters_default = [
        Parameter(
            "reaction_flux",
            "0",
            str,
            "An expression for the reaction flux in the mean field. The expression may "
            "depend on the concentration and time, which are denoted by the variables "
            "`c` and `t` respectively.",
        ),
    ]

    element_classes = (MeanfieldElement,)

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~MeanfieldActor.show_parameters` for details.
        """
        super().__init__(parameters=parameters)

        reaction_flux = self.parameters["reaction_flux"]
        self._reaction = ScalarExpression(reaction_flux, signature=["c", "t"])

    def estimate_dt(self, elements: ElementsType) -> float:
        """get the optimal time step for the simulation of the actor

        Args:
            elements (tuple of :class:`~sim.elements.fields.MeanfieldElement`):
                The element affected by the actor
        """
        s_max = np.abs(self._reaction(np.linspace(0, 1, 32), t=0)).max()
        if s_max == 0:
            return float("inf")
        else:
            # The maximum timestep for s(c) = dc/dt is based on the
            # analysis of the differential equation based on a linearized
            # expression for the flux as dc(t)/dt = k * c(t),
            # which has the solution c(t) = c(0) * exp(k * t),
            # where k is a growth or reaction rate.
            # Using an explicit Euler stepping, we find that the relative
            # error ε during a single time step of length Δt is given by
            # ε ≈ 0.5 * (k * Δt)**2 to lowest order in Δt. If we want to limit
            # ε ≤ 0.01, we thus have to choose Δt ≤ sqrt(2 * ε) / k = 0.1 / k.

            return 0.1 / s_max  # type: ignore

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray], float, float], None]:
        """return a function evolve the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.MeanfieldElement`):
                The element affected by the actor

        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float), which evolves the field_data.
        """
        reation_flux = self._reaction.get_compiled()

        @nb.jit
        def evolver(fields_data: Tuple[np.ndarray], t: float, dt: float) -> None:
            """evolve the diffusion equation explicitly"""
            (field_data,) = fields_data
            field_data += dt * reation_flux(field_data, t)

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float):
        """evolve the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.MeanfieldElement`):
                The element affected by the actor
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
        """
        (element,) = elements  # extract single element
        element.data[...] += dt * self._reaction(element.data, t)


class ScalarFieldActorBase(ActorBase, metaclass=ABCMeta):
    """base class for actors affecting discretized scalar fields"""

    element_classes = (ScalarFieldElement,)

    def estimate_dt(self, elements: ElementsType) -> float:
        """get the optimal time step for the simulation of the actor

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor

        Returns:
            float: the time step
        """
        raise NotImplementedError

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray], float, float], None]:
        """return a function evolving the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor

        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float), which evolves the field_data.
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, elements: ElementsType, t: float, dt: float):
        """evolve the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
        """
        pass


class ScalarPDEActor(ScalarFieldActorBase):
    """actor evolving a field according to a PDE"""

    def __init__(self, pde: PDEBase, parameters: Dict[str, Any] = None):
        """initialize the actor and its PDE

        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation describing the dynamics of the
                scalar field.
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~ScalarPDEActor.show_parameters` for details.
        """
        super().__init__(parameters=parameters)

        if inspect.isclass(pde):
            self._logger.warning("Got class `%s` instead of an instance", pde)
            self.pde = pde()  # type: ignore
        else:
            self.pde = pde

    @property
    def info(self) -> Dict[str, Any]:
        """dict: information about the actor"""
        result = super().info
        result["pde"] = {"class": self.pde.__class__.__name__}
        return result

    def make_evolver_numba(  # type: ignore
        self, elements: ElementsType
    ) -> Callable[[Tuple[np.ndarray], float, float], None]:
        """return a function evolving the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor

        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float), which evolves the field_data.
        """
        (element,) = elements  # extract single element
        pde_rhs = self.pde._make_pde_rhs_numba(element._field)  # type: ignore

        @jit
        def evolver(fields_data: Tuple[np.ndarray], t: float, dt: float) -> None:
            """evolve the PDE explicitly"""
            (field_data,) = fields_data
            field_data += dt * pde_rhs(field_data, t)

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float):
        """evolve the field from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
        """
        (element,) = elements  # extract single element
        rate = self.pde.evolution_rate(element._field, t)  # type: ignore
        element._field += dt * rate  # type: ignore


class DiffusionActor(ScalarPDEActor):
    """actor evolving a field according to a simple diffusion equation"""

    parameters_default = [
        Parameter(
            "diffusivity",
            1,
            float,
            "Diffusivity in the field. This actor only supports constant "
            "diffusivities. Diffusivities depending on local concentration are "
            "supported by `ReactionDiffusionActor`.",
        ),
        Parameter(
            "boundary_conditions",
            "auto_periodic_neumann",
            object,
            "Defines the boundary conditions on the field."
            + get_text_block("ARG_BOUNDARIES"),
        ),
    ]

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~DiffusionActor.show_parameters` for details.
        """
        from pde import DiffusionPDE

        # skip calling the parent init since it expects the pde, but we first
        # need to parse the parameters. We thus simply call the grand-parent
        # init method directly
        ScalarFieldActorBase.__init__(self, parameters=parameters)

        # initialize diffusion equation
        self.pde = DiffusionPDE(
            diffusivity=self.parameters["diffusivity"],
            bc=self.parameters["boundary_conditions"],
        )

    def estimate_dt(self, elements: ElementsType) -> float:
        """get the optimal time step for the simulation of the actor

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor

        Returns:
            float: the time step
        """
        (element,) = elements  # extract single element
        dx = float(element.grid.discretization.min())  # type: ignore
        return 0.1 * dx**2 / float(self.pde.diffusivity)


class ReactionDiffusionActor(ScalarPDEActor):
    """actor evolving a field according to a reaction-diffusion equation

    This class relies on the optional `phasesep` package, which needs to be
    installed separately.
    """

    parameters_default = [
        Parameter(
            "diffusivity",
            "1",
            str,
            "Diffusivity in the field. This can be an expression depending  on the "
            "local concentration that is parsed by `sympy`. Alternatively, simple "
            "numbers are also supported.",
        ),
        Parameter(
            "reaction_flux",
            "0",
            str,
            "An expression for the reaction flux in the field.",
        ),
        Parameter(
            "boundary_conditions",
            "auto_periodic_neumann",
            object,
            "Defines the boundary conditions on the field."
            + get_text_block("ARG_BOUNDARIES"),
        ),
    ]

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                Parameters affecting the actor. Call
                :meth:`~ReactionDiffusionActor.show_parameters` for details
        """
        from phasesep.pdes import ReactionDiffusionPDE

        # skip calling the parent init since it expects the pde, but we first
        # need to parse the parameters. We thus simply call the grand-parent
        # init method directly
        ScalarFieldActorBase.__init__(self, parameters=parameters)

        # initialize reaction-diffusion equation
        pde_params = {
            "diffusivity": self.parameters["diffusivity"],
            "reaction_flux": self.parameters["reaction_flux"],
            "bc": self.parameters["boundary_conditions"],
        }
        self.pde = ReactionDiffusionPDE(pde_params)

    def estimate_dt(self, elements: ElementsType) -> float:
        """get the optimal time step for the simulation of the actor

        Args:
            elements (tuple of :class:`~sim.elements.fields.ScalarFieldElement`):
                The element affected by the actor

        Returns:
            float: the time step
        """
        (element,) = elements  # extract single element

        # estimate the time step based on the chemical reaction
        if hasattr(self.pde, "_reaction"):
            # pde seems to be an instance of ReactionDiffusionPDE
            cs = np.linspace(0, 1, 32)
            s_max = np.abs(self.pde._reaction(cs, t=0)).max()
            diffusivity = self.pde.diffusivity.value
        else:
            # pde seems to be an instance of DiffusionPDE
            s_max = 0
            diffusivity = self.pde.diffusivity

        if s_max == 0:
            dt_reaction = float("inf")
        else:
            dt_reaction = 0.2 / s_max  # maximal 2% error during time step
            # This estimate is based on an analysis of the differential equation
            # dc(t)/dt = k * c(t), which has the solution c(t) = c(0) * exp(k * t), so
            # k is a growth or reaction rate. Using an explicit Euler stepping, we find
            # that the relative error ε during a single time step of length Δt is given
            # by ε ≈ 0.5 * (k * Δt)**2 to lowest order in Δt. If we want to limit
            # ε ≤ 0.02, we thus have to choose Δt ≤ sqrt(2 * ε) / k = 0.2 / k. The
            # expression is thus a conservative estimate using the maximal reaction
            # rate k = max(s).

        # estimate the time step required for diffusion
        dx = element.grid.discretization.min()  # type: ignore
        dt_diffusion = 0.1 * dx**2 / diffusivity

        return min(dt_reaction, dt_diffusion)  # type: ignore
