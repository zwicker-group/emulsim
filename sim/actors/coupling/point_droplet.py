"""
Provides an actor coupling point-like droplets to a field

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple

import numpy as np

from pde.grids.base import DimensionError
from pde.tools import spherical
from pde.tools.expressions import ScalarExpression
from pde.tools.parameters import Parameter
from pde.tools.numba import jit

from .base import CouplingActorBase
from ...elements import SphericalDropletsElement, FieldElementBase


ActorElementType = Tuple[SphericalDropletsElement, FieldElementBase]


class PointDropletActor(CouplingActorBase):
    """ actor that couples points-like droplets to a field
    
    For simplicity, these droplets interact with the field only at one point
    (their position). This approximation only works in three dimension where it
    accelerates calculations and is usually a good approximation when the
    background field varies only little on the length scale of the droplet size.
    """

    parameters_default = [
        Parameter(
            "equilibrium_concentration",
            "1e-5 / radius",
            str,
            "Expression for the equilibrium concentration. This "
            "expression can contain the variables `radius` and "
            "`position` denoting the droplet radius and its position "
            "vector, respectively. Alternatively, the value can also be "
            "an instance defining a __call__ method.",
        ),
        Parameter(
            "diffusivity",
            1.0,
            float,
            "Diffusivity in the shell surrounding the droplets",
        ),
    ]

    state_classes = (SphericalDropletsElement, FieldElementBase)

    def _parse_equilibrium_concentration(self) -> Callable:
        """ parse the expression for the equilibrium concentration
                
        Returns:
            callable: A function that can be evaluated to obtain the equilibrium
            concentration at a certain position and radius of a droplet
        """
        # parse the equilibrium concentration
        expr = self.parameters["equilibrium_concentration"]
        if callable(expr):
            # assume that the expression supports the correct syntax
            return expr  # type: ignore
        else:
            # parse the expression
            signature = [["position", "pos", "x"], ["radius", "R"], ["i", "id"]]
            return ScalarExpression(str(expr), signature, allow_indexed=True)

    def _update_cache(self, elements: ActorElementType) -> None:
        """ prepare the simulation doing pre-calculations 
        
        Args:
            elements (tuple):
                The state of all the droplets and of the field
        """
        droplets, field = elements

        if droplets.droplets.dim != field.dim:
            raise DimensionError(
                "Droplets have a different dimension than the "
                f"background ({droplets.droplets.dim} "
                f"!= {field.dim})"
            )

        self._cache["dim"] = field.dim
        self._cache["cEqOut"] = self._parse_equilibrium_concentration()

    def estimate_dt(self, elements: ActorElementType) -> float:  # type: ignore
        """ estimate the maximal time step for simulating this actor 
        
        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            float: the maximal time step
        """
        self._check_cache(elements)
        droplets, _ = elements
        D = float(self.parameters["diffusivity"])
        L = float(droplets.data["radius"].mean())
        return L ** 2 / D

    def get_flux_outside(self, radius: float, c_far: float, cEqOut: float) -> float:
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
        D = float(self.parameters["diffusivity"])

        if self._cache["dim"] == 3:
            # flux for 3d droplet without reaction
            return 4 * np.pi * D * radius * (cEqOut - c_far)  # type: ignore

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} only works in three dimensions (current "
                f"dimension is {self._cache['dim']})"
            )

    def _make_flux_outside(self) -> Callable:
        """ create a function that calculates the integrated outwards flux at
        the droplet surface given some imposed concentration value at the outer
        shell.
        
        Returns:
            callable: the function with the signature
                (radius: float, c_far: float, cEqOut: float)
                corresponding to :meth:`PointDropletActor.get_flux_outside`
        """
        D = self.parameters["diffusivity"]

        if self._cache["dim"] == 3:

            def flux_outside(radius: float, c_far: float, cEqOut: float):
                """ flux for 3d droplet without reaction """
                return 4 * np.pi * D * radius * (cEqOut - c_far)

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} only works in three dimensions (current "
                f"dimension is {self._cache['dim']})"
            )

        return flux_outside

    def get_equilibrium_concentrations(
        self, droplets: SphericalDropletsElement
    ) -> np.ndarray:
        """ returns the equilibrium concentration outside each droplet
        
        Args:
            droplets (:class:`~sim.elements.spherical_droplets.SphericalDropletsElement`):
                The state of all the droplets        
        
        Returns:
            :class:`numpy.ndarray`: The equilibrium concentration for each
                droplet with non-zero radius.
        """
        # obtain the function for calculating the equilibrium concentration
        try:
            calc_eqout = self._cache["cEqOut"]  # use cached version
        except KeyError:
            calc_eqout = self._parse_equilibrium_concentration()

        # calculate the equilibrium concentration for each droplet
        result = []
        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius > 0:
                result.append(calc_eqout(droplet.position, droplet.radius, droplet_id))

        return np.array(result)

    def _make_droplet_evolver_numba(self, elements: ActorElementType) -> Callable:
        """ create a function to evolve a single droplet from time `t` to `t + dt`
        
        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplet_data: :class:`numpy.ndarray`, droplet_id: int,
                t: float, dt: float, filed_data: :class:`numpy.ndarray`,
                field_update: :class:`numpy.ndarray`), evolving `droplet_data`
                and updating `field_update`
        """
        droplets, field = elements

        cEqOut = self._cache["cEqOut"]
        if hasattr(cEqOut, "get_compiled"):
            calc_cEqOut = cEqOut.get_compiled()
        else:
            # try compiling in case cEqOut is a function
            calc_cEqOut = jit(cEqOut)
        cBaseIn = droplets.parameters["droplet_concentration"]
        #         sBaseIn = self.pde._reaction(c=cBaseIn, t=0)

        radius = spherical.make_radius_from_volume_compiled(self._cache["dim"])
        volume = spherical.make_volume_from_radius_compiled(self._cache["dim"])

        get_concentration = field.make_get_concentration_compiled()
        add_amount = field.make_add_amount_compiled()

        calc_flux = jit(self._make_flux_outside())

        @jit(nogil=True)
        def droplet_update(
            droplet_data: np.ndarray, droplet_id: int, field_data, t: float, dt: float
        ):
            """ update a single droplet based on the surrounding field """
            R = droplet_data.radius
            V = volume(R)

            # obtain the material flux across the droplet surface
            cInf = get_concentration(field_data, droplet_data.position)
            cEqIn = cBaseIn
            cEqOut = calc_cEqOut(droplet_data.position, droplet_data.radius, droplet_id)

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
                droplet_data.radius = 0.0  # remove all droplet material
            else:
                droplet_data.radius = radius(V + dV)

            # update the scalar field at the droplet surface
            add_amount(field_data, droplet_data.position, -amount_out)

        return droplet_update  # type: ignore

    def make_evolver_numba(self, elements: ActorElementType) -> Callable:  # type: ignore
        """ return a function evolve the state from time `t` to `t + dt`
        
        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplets_data: :class:`numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(elements)

        # obtain function for updating a single droplet
        droplet_update = self._make_droplet_evolver_numba(elements)

        @jit
        def evolver(elements_data, t: float, dt: float):
            """ evolve all droplets explicitly """
            droplets_data, field_data = elements_data
            for droplet_id, droplet_data in enumerate(droplets_data):
                # skip droplets that have disappeared
                if droplet_data.radius > 0:
                    droplet_update(droplet_data, droplet_id, field_data, t, dt)

        return evolver  # type: ignore

    def evolve(self, elements: ActorElementType, t: float, dt: float) -> None:  # type: ignore
        """ evolve the state from time `t` to `t + dt`
        
        Args:
            elements (tuple):
                The state of all the droplets and of the field
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(elements)
        droplets, field = elements
        calc_cEqOut = self._cache["cEqOut"]

        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared

            # obtain the material flux across the droplet surface
            cInf = field.get_concentration(droplet.position)
            cEqIn = droplets.parameters["droplet_concentration"]
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
            field.add_amount(droplet.position, -amount_out)
