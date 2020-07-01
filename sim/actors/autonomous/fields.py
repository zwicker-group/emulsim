'''
Provides actors that influence scalar fields.

.. autosummary::
   :nosignatures:

   ~MeanfieldActor
   ~ScalarPDEActor
   ~DiffusionActor
   ~ReactionDiffusionActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


import inspect
from abc import abstractmethod, ABCMeta
from typing import Dict, Any, Callable, Type  # @UnusedImport

import numpy as np
import numba as nb

from pde.pdes.base import PDEBase
from pde.tools.numba import jit
from pde.tools.docstrings import get_text_block
from pde.tools.expressions import ScalarExpression
from pde.tools.parameters import Parameter

from .base import AutonomousActorBase
from ...elements import MeanfieldElement, ScalarFieldElement



class MeanfieldActor(AutonomousActorBase):
    """ background based on a scalar field evolving with simple diffusion """

    parameters_default = [
        Parameter('reaction_flux', '0', str,
                  "An expression for the reaction flux in the background. The "
                  "expression may depend on the concentration and time."),
    ]
    
    element_class: Type[MeanfieldElement] = MeanfieldElement


    def __init__(self, parameters: Dict[str, Any] = None):
        """ initialize the background field
        
        Args:
            parameters (dict):
                Additional parameters. Call
                :meth:`~MeanfieldField.show_parameters` for details.
        """
        super().__init__(parameters=parameters)
        
        reaction_flux = self.parameters['reaction_flux']
        try:
            from phasesep.reactions import ReactionFluxExpression
            
        except (ModuleNotFoundError, ImportError):
            # fall back to the pde package
            if str(reaction_flux) not in {'0', '0.0', 'None'}:
                raise RuntimeError('Reaction fluxes are only supported when '
                                   'the `py-phasesep` package is available') 

            # mimick the interface of ReactionFluxExpression
            self._reaction = ScalarExpression(0, signature=['c', 't'])
            self._reaction.present = False  # type: ignore
            
        else:        
            # initialize reaction flux
            self._reaction = ReactionFluxExpression(reaction_flux,
                                                    with_mu=False)
        

    def estimate_dt(self, element: MeanfieldElement) -> float:  # type: ignore
        """ estimate the time step based on the chemical reaction
        
        Args:
            element (:class:`MeanfieldElement`):
                The element of the background
        """
        s_max = np.abs(self._reaction(np.linspace(0, 1, 32), t=0)).max()
        if s_max == 0:
            return float('inf')
        else:
            return 0.1 / s_max  # type: ignore


    def make_evolver_numba(self, element: MeanfieldElement) -> Callable:  # type: ignore
        """ return a function evolve the field from time `t` to `t + dt`
        
        Args:
            element (:class:`MeanfieldElement`):
                The element of the background        

        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float, agents_data), which evolves the field_data.
        """
        reation_flux = self._reaction.get_compiled()
        
        @nb.jit
        def evolver(field_data, t: float, dt: float):
            """ evolve the diffusion equation explicitly """
            field_data += dt * reation_flux(field_data, t)

        return evolver  # type: ignore


    def evolve(self, element: MeanfieldElement, t: float, dt: float):
        """ evolve the field from time `t` to `t + dt`
        
        Args:
            element (:class:`MeanfieldElement`):
                The element of the background
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
            agents_element (:class:`agent_based.agents.base.AgentsElementBase`):
                The element of the agents (Not used by this class)
        """
        if self._reaction.present:  # type: ignore
            element.data += dt * self._reaction(element.data, t)



class ScalarFieldActorBase(AutonomousActorBase, metaclass=ABCMeta):
    """ base class for a background based on a scalar field """


    element_class: Type[ScalarFieldElement] = ScalarFieldElement
            

    def estimate_dt(self, element: ScalarFieldElement) -> float:  # type: ignore
        """ get the optimal time step for the simulation of the background
        
        Args:
            element (:class:`ScalarFieldElement`):
                The background element
                
        Returns:
            float: the time step
        """
        raise NotImplementedError


    def make_evolver_numba(self, element: ScalarFieldElement) -> Callable:  # type: ignore
        """ return a function evolve the field from time `t` to `t + dt`

        Args:
            element (:class:`ScalarFieldElement`):
                The background element

        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float, agents_data), which evolves the field_data.
        """
        raise NotImplementedError


    @abstractmethod
    def evolve(self, element: ScalarFieldElement, t: float, dt: float):
        """ evolve the field from time `t` to `t + dt`
        
        Args:
            element (:class:`MeanfieldElement`):
                The element of the background
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
            agents_element (:class:`agent_based.agents.base.AgentsElementBase`):
                The element of the agents (Not used by this class)
        """
        pass
    


class ScalarPDEActor(ScalarFieldActorBase):
    """ background based on a scalar field evolving according to a PDE """

    def __init__(self, pde: PDEBase, parameters: Dict[str, Any] = None):
        """ initialize the scalar background field and its PDE
        
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation describing the dynamics of the
                scalar background field.
            parameters (dict):
                Additional parameters. Call
                :meth:`~ScalarPDEField.show_parameters` for details.
        """
        super().__init__(parameters=parameters)
        
        if inspect.isclass(pde):
            self._logger.warning('Class `%s` has been passed instead of an '
                                 'instance.', pde)
            self.pde = pde()  # type: ignore
        else:
            self.pde = pde
        
    
    @property
    def info(self) -> Dict[str, Any]:
        """ dict: information about the background """
        result = super().info
        result['pde'] = {'class': self.pde.__class__.__name__}
        return result


    def make_evolver_numba(self, element: ScalarFieldElement) -> Callable:  # type: ignore
        """ return a function evolve the field from time `t` to `t + dt`

        Args:
            element (:class:`ScalarFieldElement`):
                The background element
                
        Returns:
            callable: A function with signature (field_data, t: float,
                dt: float, agents_data), which evolves the field_data.
        """
        pde_rhs = self.pde._make_pde_rhs_numba(element._field)

        @jit
        def evolver(field_data, t: float, dt: float):
            """ evolve the diffusion equation explicitly """
            field_data += dt * pde_rhs(field_data, t)

        return evolver  # type: ignore


    def evolve(self, element: ScalarFieldElement, t: float, dt: float):
        """ evolve the field from time `t` to `t + dt`
        
        Args:
            element (:class:`MeanfieldElement`):
                The element of the background
            t (float):
                The current time point
            dt (float):
                The time step used to evolve the element
            agents_element (:class:`agent_based.agents.base.AgentsElementBase`):
                The element of the agents (Not used by this class)
        """
        rate = self.pde.evolution_rate(element._field, t)
        element._field += dt * rate  # type: ignore



class DiffusionActor(ScalarPDEActor):
    """ background based on a scalar field evolving with simple diffusion """

    parameters_default = [
        Parameter('diffusivity', 1, float,
                  "Diffusivity in the background field. This class only "
                  "supports constant diffusivities. Diffusivities depending "
                  "on local concentration are supported by the "
                  "ReactionDiffusionPDE class."),
        Parameter('boundary_conditions', 'natural', object,
                  "Defines the boundary conditions on the background field." + 
                  get_text_block('ARG_BOUNDARIES')),
    ]


    def __init__(self, parameters: Dict[str, Any] = None):
        """ initialize the background 
        
        Args:
            parameters (dict):
                Additional parameters. Call
                :meth:`~DiffusionField.show_parameters` for details.
        """
        from pde import DiffusionPDE
        
        # skip calling the parent init since it expects the pde, but we first
        # need to parse the parameters. We thus simply call the grand-parent
        # init method directly
        ScalarFieldActorBase.__init__(self, parameters=parameters)

        # initialize diffusion equation
        self.pde = DiffusionPDE(diffusivity=self.parameters['diffusivity'],
                                bc=self.parameters['boundary_conditions'])


    def estimate_dt(self, element: ScalarFieldElement) -> float:  # type: ignore
        """ get the optimal time step for the simulation of the background
        
        Returns:
            float: the time step
        """
        dx = float(element.grid.discretization.min())
        return 0.1 * dx**2 / float(self.pde.diffusivity)



class ReactionDiffusionActor(ScalarPDEActor):
    """ scalar field background evolving with a reaction-diffusion equation
    
    This class relies on the optional `phasesep` package, which needs to be
    installed separately.
    """

    parameters_default = [
        Parameter('diffusivity', '1', str,
                  "Diffusivity in the background field. This can be an "
                  "expression that is parsed by sympy"),
        Parameter('reaction_flux', '0', str,
                  "An expression for the reaction flux in the background"),
        Parameter('boundary_conditions', 'natural', object,
                  "Defines the boundary conditions on the background field." + 
                  get_text_block('ARG_BOUNDARIES')),
    ]


    def __init__(self, parameters: Dict[str, Any] = None):
        """ initialize the background
        
        Args:
            parameters (dict):
                Additional parameters. Call
                :meth:`~ReactionDiffusionField.show_parameters` for details
        """
        from phasesep.pdes import ReactionDiffusionPDE
        
        # skip calling the parent init since it expects the pde, but we first
        # need to parse the parameters. We thus simply call the grand-parent
        # init method directly
        ScalarFieldActorBase.__init__(self, parameters=parameters)

        # initialize reaction-diffusion equation
        pde_params = {'diffusivity': self.parameters['diffusivity'],
                      'reaction_flux': self.parameters['reaction_flux'],
                      'bc': self.parameters['boundary_conditions']}
        self.pde = ReactionDiffusionPDE(pde_params)


    def estimate_dt(self, element: ScalarFieldElement) -> float:  # type: ignore
        """ get the optimal time step for the simulation of the background
        
        Args:
            element (:class:`ScalarFieldElement`):
                The background element
        
        Returns:
            float: the time step
        """
        # estimate the time step based on the chemical reaction
        if hasattr(self.pde, '_reaction'):
            # pde seems to be an instance of ReactionDiffusionPDE
            cs = np.linspace(0, 1, 32)
            s_max = np.abs(self.pde._reaction(cs, t=0)).max()
            diffusivity = self.pde.diffusivity.value
        else:
            # pde seems to be an instance of DiffusionPDE
            s_max = 0
            diffusivity = self.pde.diffusivity
            
        if s_max == 0:
            dt_reaction = float('inf')
        else:
            dt_reaction = 0.1 / s_max 
        
        # estimate the time step required for diffusion        
        dx = element.grid.discretization.min()
        dt_diffusion = 0.2 * dx**2 / diffusivity  
    
        return min(dt_reaction, dt_diffusion)  # type: ignore