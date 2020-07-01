'''
Provides the coupling of a point-like droplet to a field

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Callable

import numpy as np
import numba as nb

from pde.grids.base import DimensionError
from pde.tools import spherical, expressions
from pde.tools.parameters import Parameter
from pde.tools.numba import jit

from .base import CouplingActorBase
from ...elements import SphericalDropletsElement, FieldElementBase



class PointDropletActor(CouplingActorBase):
    """ represents points-like droplets agents
    
    For simplicity, these droplets interact with the background field only at
    one point (their position). This approximation only works in three dimension
    where it accelerates calculations and is usually a good approximation when
    the background field varies only little on the length scale of the droplet
    size.
    """

    parameters_default = [
        Parameter('equilibrium_concentration', '1e-5 / radius', str,
                  "Expression for the equilibrium concentration. This "
                  "expression can contain the variables `radius` and "
                  "`position` denoting the droplet radius and its position "
                  "vector, respectively. Alternatively, the value can also be "
                  "an instance defining a __call__ method."),
        Parameter('diffusivity', 1., float,
                  "Diffusivity in the shell surrounding the droplets")]


    state_classes = (SphericalDropletsElement, FieldElementBase)


    def _parse_equilibrium_concentration(self) -> Callable:
        """ parse the expression for the equilibrium concentration
                
        Returns:
            callable: A function that can be evaluated to obtain the equilibrium
            concentration at a certain position and radius of a droplet
        """
        # parse the equilibrium concentration
        expr = self.parameters['equilibrium_concentration']
        if callable(expr):
            # assume that the expression supports the correct syntax
            return expr  # type: ignore
        else:
            # parse the expression
            signature = [['position', 'pos', 'x'], ['radius', 'R'], ['i', 'id']]
            return expressions.ScalarExpression(str(expr), signature,
                                                allow_indexed=True)


    def _update_cache(self, droplets_state: SphericalDropletsElement,
                      background_state: FieldElementBase) -> None:
        """ prepare the simulation doing pre-calculations 
        
        Args:
            agents_state (:class:`DropletAgentsElement`):
                The state of all the droplets        
            background_state \
                   (:class:`~agent_based.backgrounds.base.FieldElementBase`):
                The state corresponding to the background
        """
        if droplets_state.droplets.dim != background_state.dim:
            raise DimensionError("Droplets have a different dimension than the "
                                 f"background ({droplets_state.droplets.dim} "
                                 f"!= {background_state.dim})")
        
        self._cache['dim'] = background_state.dim
        self._cache['cEqOut'] = self._parse_equilibrium_concentration()


    def estimate_dt(self, droplets_state: SphericalDropletsElement,  # type: ignore
                    background_state: FieldElementBase) -> float:
        """ estimate the maximal time step for simulating this agent type 
        
        Args:
            agents_state (:class:`DropletAgentsElement`):
                The state corresponding to this agent type
            background_state \
                   (:class:`~agent_based.backgrounds.base.FieldElementBase`):
                The state corresponding to the background

        Returns:
            float: the maximal time step
        """
        self._check_cache(droplets_state, background_state)
        D = float(self.parameters['diffusivity'])
        L = float(droplets_state.data['radius'].mean())
        return L**2 / D


    def get_flux_outside(self, radius: float, c_far: float, cEqOut: float) \
            -> float:
        """ returns the integrated outwards flux at the droplet surface given
        some imposed concentration value at the outer shell
        
        Args:
            radius (float):
                The current droplet radius
            c_far (float):
                The concentration at the outer side of the shell sector
            cEqOut (float):
                The concentration right at the inner side of the shell sector,
                right at the droplet surface.
            
        Returns:
            float: the integrated flux in the outward normal direction. 
        """
        D = float(self.parameters['diffusivity'])

        if self._cache['dim'] == 3:
            # flux for 3d droplet without reaction
            return 4 * np.pi * D * radius * (cEqOut - c_far)  # type: ignore

        else:
            raise NotImplementedError(f"{self.__class__.__name__} only works "
                                      "in three dimensions (current dimension "
                                      f"is {self._cache['dim']})")


    def _make_flux_outside(self) -> Callable:
        """ create a function that calculates the integrated outwards flux at
        the droplet surface given some imposed concentration value at the outer
        shell.
        
        Returns:
            callable: the function with the signature
                (radius: float, c_far: float, cEqOut: float)
                corresponding to :meth:`PointDropletAgents.get_flux_outside`
        """
        D = self.parameters['diffusivity']

        if self._cache['dim'] == 3:
            def flux_outside(radius: float, c_far: float, cEqOut: float):
                """ flux for 3d droplet without reaction """
                return 4 * np.pi * D * radius * (cEqOut - c_far)

        else:
            raise NotImplementedError(f"{self.__class__.__name__} only works "
                                      "in three dimensions (current dimension "
                                      f"is {self._cache['dim']})")

        return flux_outside


    def get_equilibrium_concentrations(self, droplets_state: SphericalDropletsElement) \
            -> np.ndarray:
        """ returns the equilibrium concentration outside each droplet
        
        Args:
            agents_state \
                (:class:`agent_based.agents.point_droplet.DropletAgentsElement`):
                The state of the agents
        
        Returns:
            :class:`numpy.ndarray`: The equilibrium concentration for each
                droplet with non-zero radius.
        """
        # obtain the function for calculating the equilibrium concentration
        try:
            calc_eqout = self._cache['cEqOut']  # use cached version
        except KeyError:
            calc_eqout = self._parse_equilibrium_concentration()
            
        # calculate the equilibrium concentration for each droplet
        result = []
        for droplet_id, droplet in enumerate(droplets_state.droplets):
            if droplet.radius > 0:
                result.append(calc_eqout(droplet.position, droplet.radius,
                                         droplet_id))
                
        return np.array(result)
        

    def _make_droplet_evolver_numba(self, droplets_state: SphericalDropletsElement,
                                    background_state: FieldElementBase) \
            -> Callable:
        """ create a function to evolve a single agent from time `t` to `t + dt`
        
        Args:
            agents_state \
                (:class:`agent_based.agents.point_droplet.DropletAgentsElement`):
                The state of all droplet agents of this instance
            background_state (:class:`FieldElementBase`):
                The state of the background

        Returns:
            callable: A function with signature
                (droplet_data: :class:`numpy.ndarray`, droplet_id: int,
                t: float, dt: float, filed_data: :class:`numpy.ndarray`,
                field_update: :class:`numpy.ndarray`), evolving `droplet_data`
                and updating `field_update`
        """        
        cEqOut = self._cache['cEqOut']
        if hasattr(cEqOut, 'get_compiled'):
            calc_cEqOut = cEqOut.get_compiled()
        else:
            # try compiling in case cEqOut is a function
            calc_cEqOut = jit(cEqOut)
        cBaseIn = droplets_state.parameters['droplet_concentration']
#         sBaseIn = self.pde._reaction(c=cBaseIn, t=0)

        radius = spherical.make_radius_from_volume_compiled(self._cache['dim'])
        volume = spherical.make_volume_from_radius_compiled(self._cache['dim'])
        
        get_concentration = background_state.make_get_concentration_compiled()
        add_amount = background_state.make_add_amount_compiled()

        calc_flux = jit(self._make_flux_outside())


        @jit(nogil=True)
        def droplet_update(droplet_data: np.ndarray, droplet_id: int,
                           background_data,
                           t: float, dt: float):                      
            """ update a single droplet based on the surrounding field """
            R = droplet_data.radius
            V = volume(R)

            # obtain the material flux across the droplet surface
            cInf = get_concentration(background_data, droplet_data.position)
            cEqIn = cBaseIn
            cEqOut = calc_cEqOut(droplet_data.position, droplet_data.radius,
                                 droplet_id)

            # Calculate the integrated fluxes at the droplet surface. The sign
            # of the fluxes is such that positive values indicate outward fluxes
            flux_out = calc_flux(R, cInf, cEqOut)

            # amount taken up from the outside per sector
            amount_out = -dt * flux_out
            
            # amount produced in the inside
            sBaseIn = 0
            amount_in = dt * sBaseIn * V

            # update the droplet volume
            dV = (amount_in + amount_out) / cEqIn
            if V + dV < 0:
                # droplet disappears
                amount_out = V * cEqIn - amount_in
                droplet_data.radius = 0.  # remove all droplet material
            else:
                droplet_data.radius = radius(V + dV)

            # update the scalar field at the droplet surface
            add_amount(background_data, droplet_data.position, -amount_out)

        return droplet_update  # type: ignore


    def make_evolver_numba(self, droplet_state: SphericalDropletsElement,  # type: ignore
                           background_state: FieldElementBase) -> Callable:
        """ return a function evolve the agents state from time `t` to `t + dt`
        
        Args:
            agents_state (:class:`DropletAgentsElement`):
                The state of all the droplets        
            background_state (:class:`FieldElementBase`):
                The state of the background

        Returns:
            callable: A function with signature
                (agents_data: :class:`numpy.ndarray`, t: float, dt: float,
                background_data), evolving `agents_data`
        """
        self._check_cache(droplet_state, background_state)
        
        # obtain function for updating a single droplet
        droplet_update = self._make_droplet_evolver_numba(droplet_state,
                                                          background_state)
        
        # obtain the signature for the evolver
        dr_type = nb.typeof(droplet_state.data) 
        bg_type = nb.typeof(background_state.data)

        @jit(signature=nb.void(dr_type, bg_type, nb.float64, nb.float64),
             nogil=True)
        def evolver(agents_data: np.ndarray, background_data,
                    t: float, dt: float):
            """ evolve all agents explicitly """
            for droplet_id, droplet_data in enumerate(agents_data):
                # skip droplets that have disappeared
                if droplet_data.radius > 0:
                    droplet_update(droplet_data, droplet_id, background_data,
                                   t, dt)

        return evolver  # type: ignore


    def evolve(self, droplet_state: SphericalDropletsElement,  # type: ignore
               background_state: FieldElementBase,
               t: float, dt: float) -> None:
        """ evolve the agents state from time `t` to `t + dt`
        
        Args:
            agents_state (:class:`DropletAgentsElement`):
                The state of all agents described by this instance
            background_state (:class:`FieldElementBase`):
                The state of the background
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(droplet_state, background_state)
        calc_cEqOut = self._cache['cEqOut']

        for droplet_id, droplet in enumerate(droplet_state.droplets):
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared

            # obtain the material flux across the droplet surface
            cInf = background_state.get_concentration(droplet.position)
            cEqIn = droplet_state.parameters['droplet_concentration']
            cEqOut = calc_cEqOut(droplet.position, droplet.radius, droplet_id)

            # Calculate the integrated fluxes at the droplet surface. The sign
            # of the fluxes is such that positive values indicate outward fluxes
            flux_out = self.get_flux_outside(droplet.radius, cInf, cEqOut)
            # amount taken up from the outside per shell
            amount_out = -dt * flux_out
            # amount produced inside the droplet
            # amount_total_in = params['dt'] * sBaseIn * droplet.volume
            amount_in = 0

            # update the droplet volume
            dV = (amount_in + amount_out) / cEqIn
            if droplet.volume + dV < 0:
                # make sure
                amount_out = droplet.volume * cEqIn - amount_in
                droplet.volume = 0  # remove all droplet material
            else:
                droplet.volume = droplet.volume + dV

            # update the scalar field at the droplet boundary
            background_state.add_amount(droplet.position, -amount_out)
