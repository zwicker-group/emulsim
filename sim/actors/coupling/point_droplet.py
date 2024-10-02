"""Provides an actor coupling point-like droplets to a field.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from droplets.tools import spherical
from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit

from ... import Parameter
from ...elements import FieldElementBase, ReservoirElement, SphericalDropletsElement
from ..base import ActorBase

ActorElementType = tuple[SphericalDropletsElement, FieldElementBase]


class PointDropletActor(ActorBase):
    """Actor that couples points-like droplets to a field.

    For simplicity, these droplets interact with the field only at one point (their
    position). For a physical correct model, where the exchange flux with the dilute
    phase is governed by diffusion, this approximation only works in three dimension
    where it accelerates calculations and is usually a good approximation when the
    background field varies only little on the length scale of the droplet size. To
    cover also other dimension, a simple linear exchange flux model is also supported.
    """

    parameters_default = [
        Parameter(
            "equilibrium_concentration",
            "1e-5 / radius",
            object,
            "Expression/function determining the equilibrium concentration. This can "
            "be a mathematical expression as a string or a python function. The "
            "expression can contain the variables `position`, `radius`, and `id` "
            "denoting the droplet radius, its position vector, and its identity (the "
            "index in the list of droplets), respectively. If the value is a function "
            "it should have the signature (position, radius, i) and return the "
            "equilibrium concentration. it can also be an instance defining a __call__ "
            "method that returns the equilibrium concentration and a `get_compiled` "
            "method that returns a numba compiled function for calculating it."
            "(position, radius, i).",
        ),
        Parameter(
            "flux_model",
            "diffusion",
            str,
            "Specifies how the exchange of material between the droplet and the dilute "
            "phase is modeled. Possible options: 1) 'diffusion', which uses a real "
            "physical diffusion model. The parameter `diffusivity` determines rate and "
            "the model is only applicable in 3 dimensional systems. 2) 'linear', which "
            "assumes that the flux is a linear function of the difference between the "
            "equilibrium concentration `equilibrium_concentration` and the actually "
            "measured concentration. The parameter `exchange_rate` defines the "
            "pre-factor governing the rate of the exchange.",
        ),
        Parameter(
            "diffusivity",
            1.0,
            float,
            "Diffusivity of the droplet material in the dilute phase, which determines "
            "the rate if `flux_model` is `diffusion`.",
        ),
        Parameter(
            "exchange_rate",
            1.0,
            object,
            "Rate with which droplet material is exchanged with the dilute phase when "
            "the `flux_model` is `linear`. Must be the total flux exchanged from the "
            "droplet toward the background field. This can either be a number of an "
            "expression, which can depend on the radius `R` of the droplet.",
        ),
        Parameter(
            "mean_reaction_inside",
            "0",
            str,
            "Mean reaction rate inside the droplet, which determines the production of "
            "droplet material per unit volume. This can be an expression that depends "
            "on the droplet radius `R`, its location `position`, or its identity `id` "
            "(the index in the list of droplets). Use negative values to destroy "
            "droplet material inside the droplet.",
        ),
    ]

    element_classes = (SphericalDropletsElement, [ReservoirElement, FieldElementBase])  # type: ignore

    def _parse_expression(
        self, expr: Callable | str | float, signature: list[list[str]]
    ) -> Callable:
        """Parse an expression for a parameter.

        Returns:
            callable: A function that can be evaluated to obtain the expression
        """
        if callable(expr):
            # assume that the expression supports the correct syntax
            return expr
        else:
            # parse the expression
            return ScalarExpression(str(expr), signature, allow_indexed=True)

    def _update_cache(self, elements: ActorElementType) -> None:
        """Prepare the simulation doing pre-calculations.

        Args:
            elements (tuple):
                The state of all the droplets and of the field
        """
        droplets, field = elements

        if droplets.dim is not None:
            field.check_coupling_dim(droplets.dim)
        self._cache["dim"] = droplets.dim

        self._cache["cEqOut"] = self._parse_expression(
            self.parameters["equilibrium_concentration"],
            [["position", "pos", "x"], ["radius", "R"], ["i", "id"]],
        )
        self._cache["exchange_rate"] = self._parse_expression(
            self.parameters["exchange_rate"], [["radius", "R"]]
        )
        self._cache["sBaseIn"] = self._parse_expression(
            self.parameters["mean_reaction_inside"],
            [
                ["position", "pos", "x"],
                ["radius", "R"],
                ["i", "id"],
            ],
        )

    def estimate_dt(self, elements: ActorElementType) -> float:  # type: ignore
        """Estimate the maximal time step for simulating this actor.

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            float: the maximal time step
        """
        self._check_cache(elements)
        droplets, field = elements
        mean_radius = float(droplets.data["radius"].mean())

        if self.parameters["flux_model"] == "diffusion":
            # estimate time scale from diffusion across droplet
            D = float(self.parameters["diffusivity"])
            dt: float = 0.1 * mean_radius**2 / D

        elif self.parameters["flux_model"] == "linear":
            # estimate time scale from dissolution of droplet by flux
            calc_exchange_rate = self._cache["exchange_rate"]
            exchange_rate = calc_exchange_rate(mean_radius)
            ceqs = self.get_equilibrium_concentrations(droplets)
            if len(ceqs) == 0:
                dt = math.nan  # cannot determine time step
            else:
                c0 = field.average_concentration
                flux = exchange_rate * abs(ceqs.mean() - c0)
                if droplets.droplet_count > 0 and flux > 0:
                    mean_amount = droplets.total_amount / droplets.droplet_count
                    dt = float(1e-3 * mean_amount / flux)
                else:
                    dt = math.nan  # cannot determine time step

        else:
            raise NotImplementedError(
                "Flux model `{}` undefined".format(self.parameters["flux_model"])
            )
        return dt

    def get_flux_outside(self, radius: float, c_back: float, cEqOut: float) -> float:
        """Returns the integrated outwards flux at the droplet surface given some
        imposed concentration value far away.

        Args:
            radius (float):
                The current droplet radius
            c_back (float):
                The concentration in the background field at the position of the droplet
            cEqOut (float):
                The concentration right at the droplet surface.

        Returns:
            float: the integrated flux in the outward normal direction.
        """
        if self.parameters["flux_model"] == "diffusion":
            D = float(self.parameters["diffusivity"])

            if self._cache["dim"] == 3:
                # flux for 3d droplet without reaction
                return 4 * np.pi * D * radius * (cEqOut - c_back)

            else:
                raise NotImplementedError(
                    "Flux model `diffusion` only works in three dimensions "
                    f"(current dimension is {self._cache['dim']}). Use flux model "
                    "`linear` instead."
                )

        elif self.parameters["flux_model"] in {"linear", "expression"}:
            calc_exchange_rate = self._cache["exchange_rate"]
            return calc_exchange_rate(radius) * (cEqOut - c_back)  # type: ignore

        else:
            raise NotImplementedError(
                "Flux model `{}` undefined".format(self.parameters["flux_model"])
            )

    def _make_flux_outside(self) -> Callable:
        """Create a function that calculates the integrated outwards flux at the droplet
        surface given some imposed concentration far away.

        Returns:
            callable: The function with the signature (radius: float, c_back: float,
            cEqOut: float) corresponding to :meth:`PointDropletActor.get_flux_outside`
        """
        if self.parameters["flux_model"] == "diffusion":
            D = self.parameters["diffusivity"]

            if self._cache["dim"] == 3:

                def flux_outside(radius: float, c_back: float, cEqOut: float):
                    """Diffusive exchange flux for 3d droplet."""
                    return 4 * np.pi * D * radius * (cEqOut - c_back)

            else:
                raise NotImplementedError(
                    "Flux model `diffusion` only works in three dimensions "
                    f"(current dimension is {self._cache['dim']})"
                )

        elif self.parameters["flux_model"] in {"linear", "expression"}:
            exchange_rate = self._cache["exchange_rate"]
            if hasattr(exchange_rate, "get_compiled"):
                calc_exchange_rate = exchange_rate.get_compiled()
            else:
                # try compiling in case exchange_rate is a function
                calc_exchange_rate = jit(exchange_rate)

            def flux_outside(radius: float, c_back: float, cEqOut: float):
                """Linear exchange flux."""
                return calc_exchange_rate(radius) * (cEqOut - c_back)

        else:
            raise NotImplementedError(
                "Flux model `{}` undefined".format(self.parameters["flux_model"])
            )

        return flux_outside

    def get_equilibrium_concentrations(
        self, droplets: SphericalDropletsElement
    ) -> np.ndarray:
        """Returns the equilibrium concentration outside each droplet.

        Args:
            droplets (:class:`~sim.elements.spherical_droplets.SphericalDropletsElement`):
                The state of all the droplets

        Returns:
            :class:`~numpy.ndarray`: The equilibrium concentration for each
                droplet with non-zero radius.
        """
        # obtain the function for calculating the equilibrium concentration
        try:
            calc_eqout = self._cache["cEqOut"]  # use cached version
        except KeyError:
            calc_eqout = self._parse_expression(
                self.parameters["equilibrium_concentration"],
                [["position", "pos", "x"], ["radius", "R"], ["i", "id"]],
            )

        # calculate the equilibrium concentration for each droplet
        result = []
        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius > 0:
                ceq = calc_eqout(droplet.position, droplet.radius, droplet_id)
                if np.isfinite(ceq):
                    result.append(ceq)

        return np.array(result)

    def _make_droplet_evolver_numba(
        self, elements: ActorElementType
    ) -> Callable[[tuple[np.ndarray], int, np.ndarray, float, float], None]:
        """Create a function to evolve a single droplet from time `t` to `t + dt`

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplet_data: :class:`~numpy.ndarray`, droplet_id: int,
                t: float, dt: float, filed_data: :class:`~numpy.ndarray`,
                field_update: :class:`~numpy.ndarray`), evolving `droplet_data`
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
        sBaseInFunc = self._cache["sBaseIn"]
        if hasattr(sBaseInFunc, "get_compiled"):
            calc_sBaseIn = sBaseInFunc.get_compiled()
        else:
            # try compiling in case sBaseIn is a function
            calc_sBaseIn = jit(sBaseInFunc)

        radius = spherical.make_radius_from_volume_compiled(self._cache["dim"])
        volume = spherical.make_volume_from_radius_compiled(self._cache["dim"])

        get_concentration = field.make_get_concentration_compiled()
        add_amount = field.make_add_amount_compiled()

        calc_flux = jit(self._make_flux_outside())

        @jit(nogil=True)
        def droplet_update(
            droplet_data: np.recarray,
            droplet_id: int,
            field_data: np.ndarray,
            t: float,
            dt: float,
        ) -> None:
            """Update a single droplet based on the surrounding field."""
            R = droplet_data.radius

            # obtain the material flux across the droplet surface
            cEqOut = calc_cEqOut(droplet_data.position, droplet_data.radius, droplet_id)
            if np.isinf(cEqOut):
                return  # skip this droplet
            cEqIn = cBaseIn
            cInf = get_concentration(field_data, droplet_data.position)

            # Calculate the integrated fluxes exchanged between the droplet and the
            # background. The sign is such that positive values indicate outward fluxes
            flux_out = calc_flux(R, cInf, cEqOut)

            # amount taken up from the outside per sector
            amount_out = -dt * flux_out

            # amount produced in the inside
            sBaseIn = calc_sBaseIn(
                droplet_data.position, droplet_data.radius, droplet_id
            )
            V = volume(R)
            amount_in = dt * sBaseIn * V

            # update the droplet volume
            dV = (amount_in + amount_out) / cEqIn
            if V + dV < 0:
                # droplet disappears
                amount_out = -V * cEqIn - amount_in
                droplet_data.radius = 0.0  # remove all droplet material
            else:
                droplet_data.radius = radius(V + dV)

            # update the scalar field at the droplet position
            add_amount(field_data, droplet_data.position, -amount_out)

        return droplet_update  # type: ignore

    def make_evolver_numba(  # type: ignore
        self, elements: ActorElementType
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """Return a function evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(elements)

        # obtain function for updating a single droplet
        droplet_update = self._make_droplet_evolver_numba(elements)

        @jit
        def evolver(
            elements_data: tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """Evolve all droplets explicitly."""
            droplets_data, field_data = elements_data
            for droplet_id, droplet_data in enumerate(droplets_data):
                # skip droplets that have disappeared
                if droplet_data.radius > 0:
                    droplet_update(droplet_data, droplet_id, field_data, t, dt)

        return evolver  # type: ignore

    def evolve(self, elements: ActorElementType, t: float, dt: float) -> None:  # type: ignore
        """Evolve the state from time `t` to `t + dt`

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
        calc_sBaseIn = self._cache["sBaseIn"]

        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared

            # obtain the material flux across the droplet surface
            cEqOut = calc_cEqOut(droplet.position, droplet.radius, droplet_id)
            if np.isinf(cEqOut):
                continue
            cEqIn = droplets.parameters["droplet_concentration"]
            cInf = field.get_concentration(droplet.position)

            # Calculate the integrated fluxes exchanged between the droplet and the
            # background. The sign is such that positive values indicate outward fluxes
            flux_out = self.get_flux_outside(droplet.radius, cInf, cEqOut)
            # amount taken up from the outside
            amount_out = -dt * flux_out
            # amount produced inside the droplet
            sBaseIn = calc_sBaseIn(droplet.position, droplet.radius, droplet_id)
            amount_in = dt * sBaseIn * droplet.volume

            # update the droplet volume
            dV = (amount_in + amount_out) / cEqIn
            if droplet.volume + dV < 0:
                # make sure
                amount_out = -droplet.volume * cEqIn - amount_in
                droplet.volume = 0  # remove all droplet material
            else:
                droplet.volume = droplet.volume + dV

            # update the scalar field at the droplet position
            field.add_amount(droplet.position, -amount_out)
