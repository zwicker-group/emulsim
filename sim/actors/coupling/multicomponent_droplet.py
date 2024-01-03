"""
Provides an actor coupling multicomponent droplets to background fields

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numba as nb
import numpy as np

from droplets.tools import spherical
from pde import ScalarField
from pde.fields.base import FieldBase
from pde.grids.base import DimensionError
from pde.tools.docstrings import get_text_block
from pde.tools.expressions import TensorExpression
from pde.tools.numba import jit

from ... import Parameter
from ...elements import FieldCollectionElement, MulticomponentDropletsElement
from ..base import ActorBase

ActorElementType = Tuple[MulticomponentDropletsElement, FieldCollectionElement]


class SolventFractionError(RuntimeError):
    """error indicating that the solvent fraction was not in [0, 1]"""

    pass


def _make_regularizer(
    num_comps: int, eps: float = 1e-8
) -> Callable[[np.ndarray], float]:
    """create function regularizing compositions

    Args:
        num_comps (int):
            Number of components to regularize for
        eps (float):
            Minimal deviation from boundaries of interval [0, 1]
    """
    vmin = 0.0 + eps
    vmax = 1.0 - eps
    sum_max = 1.0 - eps
    assert num_comps * vmin < sum_max
    sum_eps_max = sum_max - num_comps * vmin

    @jit
    def regularize(phi: np.ndarray) -> np.ndarray:
        """regularize a state ensuring variables stay within bounds"""
        correction = np.zeros(num_comps)

        if phi.ndim == 1:
            # a single set of concentrations is given

            # adjust each variable individually
            for i in range(num_comps):
                if phi[i] < vmin:
                    correction[i] += vmin - phi[i]
                    phi[i] = vmin
                elif phi[i] > vmax:
                    correction[i] += phi[i] - vmax
                    phi[i] = vmax

            # limit the sum of all variables
            if np.isfinite(sum_max):
                eps_sum = 0.0
                for i in range(num_comps):
                    eps_sum += phi[i] - vmin
                if eps_sum > sum_eps_max:
                    factor = sum_eps_max / eps_sum
                    for i in range(num_comps):
                        phi[i] = vmin + factor * (phi[i] - vmin)

            return correction

        else:
            # an array of concentrations is given

            # adjust each variable individually
            for i in range(num_comps):
                for j in range(phi[0].size):
                    if phi[i].flat[j] < vmin:
                        correction[i] += vmin - phi[i].flat[j]
                        phi[i, ...].flat[j] = vmin
                    elif phi[i].flat[j] > vmax:
                        correction[i] += phi[i].flat[j] - vmax
                        phi[i, ...].flat[j] = vmax

            # limit the sum of all variables
            if np.isfinite(sum_max):
                for j in range(phi[0].size):
                    eps_sum = 0.0
                    for i in range(num_comps):
                        eps_sum += phi[i].flat[j] - vmin
                    if eps_sum > sum_eps_max:
                        factor = sum_eps_max / eps_sum
                        for i in range(num_comps):
                            new_value = vmin + factor * (phi[i].flat[j] - vmin)
                            phi[i, ...].flat[j] = new_value

            # Note that we needed to use phi[i, ...] to write to the array also
            # when it is 1d to circumvent a known bug:
            # https://github.com/numpy/numpy/issues/16881

        return correction

    return regularize  # type: ignore


class MulticomponentDropletActor(ActorBase):
    """actor that couples points-like multicomponent droplets to multiple field

    For simplicity, these droplets interact with the field only at one point (their
    position) using a simple linear exchange flux model. This model can be derived in
    the simple case of a Cahn-Hilliard equation with a mobility that scales with the
    fraction. The model is currently restricted to 1D and 3D systems.

    The system describes :math:`N` interacting components that are embedded in a
    solvent. The solvent is not described explicitly, but rather derived from the
    incompressibility condition.
    """

    parameters_default = [
        Parameter(
            "chis",
            np.zeros((1, 1)),
            np.array,
            "Interaction parameters between all described components. This parameter "
            "also determines the number of described components",
        ),
        Parameter(
            "chis_solvent",
            0,
            np.array,
            "Interaction parameters between described components and the solvent",
        ),
        Parameter(
            "reactions",
            None,
            object,
            "Function or expression to specify reactions in the system",
        ),
        Parameter(
            "surface_tension",
            0.0,
            float,
            "Surface tension that determines the Laplace pressure, e.g., the additional "
            "pressure inside the droplets.",
        ),
        Parameter(
            "mobility",
            1.0,
            np.array,
            "Diffusive transport coefficients. This factor determines the diffusivities "
            "of molecules in the dilute phase and thus how fast droplets change size. "
            "The corresponding Onsager coefficient is the product of these mobilities "
            "and the fraction of the fields. A single number implies sets the same "
            "mobility for all components.",
        ),
        Parameter(
            "volume_relaxation_factor",
            1,
            float,
            "This factor affects the relaxation of the volume as a response to "
            "pressure gradients. Since volume relaxation is dominated by solvent "
            "exchange, this factor can be interpreted as the ratio of the solvent "
            "mobility to the mobility of all other components. However, large values "
            "can lead to numerical instabilities.",
        ),
        Parameter(
            "boundary_conditions",
            "auto_periodic_neumann",
            object,
            "Defines the boundary conditions on the field."
            + get_text_block("ARG_BOUNDARIES"),
        ),
        Parameter(
            "dissolve_radius",
            0.5,
            float,
            "Minimal radius before a droplet is considered dissolved. This cutoff is "
            "necessary since very small droplets can lead to numerical instabilities "
            "where the composition is no longer within [0, 1].",
        ),
        Parameter(
            "dissolve_fraction",
            1e-6,
            float,
            "Minimal total fraction in a droplet before it is considered dissolved. "
            "This threshold ensures that large droplets that have the same composition "
            "as the background are removed.",
        ),
        Parameter(
            "min_fraction",
            1e-8,
            float,
            "Minimal value fraction any fraction may attain. Fractions must not become "
            "zero since otherwise the logarithms appearing in the entropic "
            "contributions cannot be evaluated. The minimal fraction is strictly "
            "enforced, which can potentially lead to mass loss during a simulation. To "
            "control for this, we record the accumulated corrections applied to each "
            "component in the array `diagnostics['amount_corrections']`",
        ),
    ]

    element_classes = (MulticomponentDropletsElement, FieldCollectionElement)

    @classmethod
    def from_linear_reactions(
        cls,
        parameters: dict[str, Any],
        rates: np.ndarray,
        production: np.ndarray | None = None,
    ) -> MulticomponentDropletActor:
        """create functions suitable to describe linear reactions

        Args:
            parameters (dict):
                Parameters defining the behavior of the actor. Call
                :meth:`~ActorBase.show_parameters` for details.
            rates (:class:`~numpy.ndarray`):
                The rate matrix describing the conversion of all components
            production (:class:`~numpy.ndarray`, optional):
                The zeroth-order production flux of all components

        Returns:
            callable: a function that determines the reaction rates or `None` if no
            reactions are present (i.e., all inputs are zero)
        """
        if "reactions" in parameters:
            raise ValueError("Cannot use parameter `reactions` and explicit reactions")

        rate_matrix = np.asarray(rates)
        if rate_matrix.ndim == 1:
            rate_matrix = np.diag(rate_matrix)
        num_comps = len(rate_matrix)

        if production is None:
            production_rate = np.zeros(num_comps)
        else:
            production_rate = np.broadcast_to(production, (num_comps,))

        if np.allclose(rate_matrix, 0) and np.allclose(production_rate, 0):
            parameters["reactions"] = None

        else:

            def droplet_reactions(
                phis: np.ndarray,
                mus: np.ndarray,
                t: float,
                out: np.ndarray | None = None,
            ) -> np.ndarray:
                """function implementing the linear reactions"""
                if out is None:
                    out = np.empty_like(phis)
                for i in range(num_comps):
                    out[i] = production_rate[i]
                    for j in range(num_comps):
                        out[i] += rate_matrix[i, j] * phis[j]
                return out

            parameters["reactions"] = droplet_reactions

        return cls(parameters)

    @property
    def chis_full(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the full interaction matrix including solvent"""
        chis = self.parameters["chis"]
        num_comps = len(chis)
        result = np.zeros((num_comps + 1, num_comps + 1))
        result[:num_comps, :num_comps] = chis
        chi_solvent = self.parameters["chis_solvent"]
        result[:num_comps, -1] = result[-1, :num_comps] = chi_solvent
        return result

    @property
    def _chis_solvent(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: interactions between the components and solvent"""
        chis_sol = self.parameters["chis_solvent"]
        num_comps = len(self.parameters["chis"])
        return np.broadcast_to(chis_sol, (num_comps,)).astype(float)

    @property
    def _chis_reduced(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: reduced interaction matrix with solvent-effects"""
        chis = self.parameters["chis"]
        chis_sol = self._chis_solvent
        return chis - chis_sol - chis_sol.reshape(-1, 1)  # type: ignore

    def _make_calc_state_vars(
        self,
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray, float]]:
        """create function calculating the state variables

        Returns:
            A function that calculates free energy, chemical potentials and pressure for
            a given composition
        """
        num_comps = len(self.parameters["chis"])
        chis_sol = self._chis_solvent
        chis_red = self._chis_reduced
        assert chis_sol.shape == (num_comps,)
        assert chis_red.shape == (num_comps, num_comps)

        @jit
        def calc_state_vars(phis: np.ndarray) -> tuple[float, np.ndarray, float]:
            """calculates thermodynamic state variables from composition"""
            assert phis.shape == (num_comps,)
            phi_sol = 1.0 - phis.sum(axis=0)
            if phi_sol <= 0:
                raise SolventFractionError("Negative solvent concentration in droplet")

            log_phi_sol = np.log(phi_sol)
            f = phi_sol * log_phi_sol  # entropy of solvent
            mu = np.full(num_comps, -log_phi_sol)  # chemical potentials
            p = 0.0  # pressure
            for i in range(num_comps):  # iterate components
                f += phis[i] * np.log(phis[i])  # entropy of component i
                f += chis_sol[i] * phis[i]  # captures part of interaction with solvent
                mu[i] += np.log(phis[i]) + chis_sol[i]
                for j in range(num_comps):
                    f += 0.5 * chis_red[i, j] * phis[i] * phis[j]
                    mu[i] += chis_red[i, j] * phis[j]
                p += phis[i] * mu[i]
            p -= f

            return f, mu, p

        return calc_state_vars  # type: ignore

    def _update_cache(self, elements: ActorElementType) -> None:
        """prepare the simulation doing pre-calculations

        Args:
            elements (tuple):
                The state of all the droplets and of the field
        """
        droplets_el, fields_el = elements

        # check spatial dimension
        if droplets_el.dim is None:
            if fields_el.dim is None:
                dim = 1  # fall back to simple choice
            else:
                dim = fields_el.dim
        else:
            dim = droplets_el.dim
            if fields_el.dim is not None and fields_el.dim != dim:
                raise DimensionError(
                    "Droplets have a different dimension than the background "
                    f"({droplets_el.dim} != {fields_el.dim})"
                )

        # check number of components
        num_comps = len(self.parameters["chis"])
        if droplets_el.num_comps is not None and droplets_el.num_comps != num_comps:
            raise RuntimeError(
                "Droplets need as many components as specified in interaction matrix "
                f"({droplets_el.num_comps} != {num_comps})"
            )
        if fields_el.num_fields is not None and fields_el.num_fields != num_comps:
            raise RuntimeError(
                "Fields need as many components as specified in interaction matrix "
                f"({fields_el.num_fields} != {num_comps})"
            )

        # TODO: add a check whether the fractions inside the droplet + the outside
        # add up

        # determine basic quantities and fall back to simple choices when empty
        self._cache["dim"] = dim
        self._cache["num_comps"] = num_comps
        Rmin = self.parameters["dissolve_radius"]
        self._cache["volume_min"] = spherical.volume_from_radius(Rmin, dim)
        self._cache["calc_state_vars"] = self._make_calc_state_vars()
        self._cache["interpolate_fields"] = fields_el.make_get_concentrations_compiled()
        min_fraction = self.parameters["min_fraction"]
        self._cache["regularize"] = _make_regularizer(num_comps, eps=min_fraction)

        # check reactions
        if self.parameters["reactions"] is None:
            # no reactions
            def noop(phis: np.ndarray, mus: np.ndarray, t: float) -> np.ndarray:
                return np.zeros_like(phis)

            self._cache["has_reaction"] = False
            self._cache["reaction_flux"] = noop

        elif callable(self.parameters["reactions"]):
            # callable function is given
            self._cache["has_reaction"] = True
            self._cache["reaction_flux"] = jit(self.parameters["reactions"])

        else:
            # assume an expression is given
            expr = TensorExpression(self.parameters["reactions"], ["phi", "mu", "t"])
            self._cache["has_reaction"] = True
            self._cache["reaction_flux"] = expr.get_compiled_array(single_arg=False)

    def get_thermodynamic_quantity(
        self,
        droplets: MulticomponentDropletsElement,
        fields: FieldCollectionElement,
        kind: str,
    ) -> tuple[np.ndarray, FieldBase]:
        """return a thermodynamic quantity in the droplets and the background field

        Args:
            droplets (:class:`MulticomponentDropletsElement`):
                The element describing all the droplets
            fields (:class:`FieldCollectionElement`):
                The element describing all the background fields
            kind (str):
                Determines which quantity to return. Possible choices are
                "free energy density", "chemical potential", and "pressure"

        Returns:
            tuple of :class:`~numpy.ndarray` (selected quantity for each droplet) and
            :class:`~pde.fields.base.FieldBase` (selected quantity for the background).
        """
        data_kinds = {"free energy": 0, "chemical potential": 1, "pressure": 2}
        data_id = data_kinds[kind.lower()]
        try:
            calc_state_vars = self._cache["calc_state_vars"]
        except KeyError:
            calc_state_vars = self._make_calc_state_vars()
            self._cache["calc_state_vars"] = calc_state_vars

        # determine data in all droplets
        data_droplets = []
        for droplet in droplets.droplets:
            if droplet.radius > 0:
                phis_out = fields.get_concentrations(droplet.position)
                phis_in = droplet.phis + phis_out  # raise above background
                data_droplets.append(calc_state_vars(phis_in)[data_id])

        # determine data in the background field
        if data_id == 1:
            data_field: FieldBase = fields.field.copy(label=kind)
        else:
            data_field = ScalarField(fields.grid, label=kind)
        for cell_id in np.ndindex(*fields.grid.shape):
            idx = (...,) + cell_id
            data_field.data[idx] = calc_state_vars(fields.data[idx])[data_id]

        return np.array(data_droplets), data_field

    def get_droplet_fractions(self, elements: ActorElementType) -> np.ndarray:
        """calculates the fractions outside and inside of all droplets

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            :class:`~numpy.ndarray`: the fractions outside and inside of all droplets
            for all components. This array has shape (num_drops, 2, num_comps), where
            the second axis distinguishes outside and inside.
        """
        droplets_el, fields_el = elements
        result: list[list[np.ndarray] | np.ndarray] = []
        for droplet in droplets_el.droplets:
            if droplet.radius > 0:
                phi_out = fields_el.get_concentrations(droplet.position)
                phi_in = droplet.phis + phi_out  # raise above background
                result.append([phi_out, phi_in])
            else:
                result.append(np.full((2, droplet.num_comps), np.nan))
        return np.array(result)

    def make_evolver_numba(  # type: ignore
        self, elements: ActorElementType
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data, t: float,
                dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(elements)
        _, fields_el = elements

        # obtain constants that need to be used
        dim = self._cache["dim"]
        num_comps = self._cache["num_comps"]
        volume_relaxation_factor = self.parameters["volume_relaxation_factor"]
        mobility = np.broadcast_to(self.parameters["mobility"], (num_comps,))
        surface_tension = self.parameters["surface_tension"]
        phi_min = self.parameters["dissolve_fraction"]
        volume_min = self._cache["volume_min"]
        min_fraction = self.parameters["min_fraction"]
        has_reaction = self._cache["has_reaction"]
        chis_solvent = self._chis_solvent
        chis_reduced = self._chis_reduced

        # obtain functions that need to be used
        regularize = self._cache["regularize"]
        interpolate_fields = self._cache["interpolate_fields"]
        bcs = self.parameters["boundary_conditions"]
        laplace = fields_el.grid.make_operator("laplace", bcs)
        calc_state_vars = self._cache["calc_state_vars"]
        radius = spherical.make_radius_from_volume_compiled(self._cache["dim"])
        volume = spherical.make_volume_from_radius_compiled(self._cache["dim"])
        add_amounts = fields_el.make_add_amounts_compiled()
        reaction_flux = self._cache["reaction_flux"]

        self.diagnostics.setdefault("amount_corrections", np.zeros(num_comps))

        @jit
        def evolver(
            elements_data: tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """evolve all droplets and the fields explicitly"""
            droplets_data, fields_data = elements_data

            # determine diffusive flux in the background
            j_back = np.empty_like(fields_data)
            for i in range(num_comps):
                j_back[i] = -mobility[i] * laplace(fields_data[i])

            # determine reaction flux in the background
            if has_reaction:
                phi_back = fields_data
                mu_back = np.empty_like(fields_data)
                for i in range(num_comps):
                    mu_back[i] = np.log(phi_back[i]) + chis_solvent[i]
                    for j in range(num_comps):
                        mu_back[i] += chis_reduced[i, j] * phi_back[j]
                s_out = reaction_flux(phi_back, mu_back, t)

            # update all droplets
            amount_corrections = np.zeros(num_comps)
            for droplet_data in droplets_data:
                if droplet_data.radius <= 0:
                    continue  # skip droplets that have disappeared

                # read basic properties of the droplet
                R = droplet_data.radius
                V = volume(R)
                amounts = droplet_data.amounts

                # determine the compositions inside and outside
                phi_out = interpolate_fields(fields_data, droplet_data.position)
                phi_in = amounts / V + phi_out
                amount_corrections += V * (regularize(phi_out) + regularize(phi_in))

                # obtain the material flux across the droplet surface
                _, mu_in, p_in = calc_state_vars(phi_in)
                _, mu_out, p_out = calc_state_vars(phi_out)

                if has_reaction:
                    # determine reaction fluxes inside droplet and in the
                    # corresponding background zone
                    s_in = reaction_flux(phi_in, mu_in, t)
                    Sin = dt * V * s_in
                    Sout = dt * V * interpolate_fields(s_out, droplet_data.position)
                else:
                    # there are no reactions
                    Sin = Sout = np.zeros(num_comps)

                # Laplace pressure is exerted onto the droplet
                p_laplace = (dim - 1) * surface_tension / R

                # dynamics fluxes as linear functions of the respective forces
                if dim == 1:
                    vol_step = dt * mobility.mean() * volume_relaxation_factor
                    diff_step = dt * mobility * phi_out
                elif dim == 3:
                    factor = dt * 4 * np.pi * R * mobility
                    vol_step = factor.mean() * volume_relaxation_factor
                    diff_step = factor * phi_out
                else:
                    raise NotImplementedError("Only implemented for dim ∈ [1, 3]")
                # droplet volume increases as response to pressure difference
                ΔV = vol_step * (p_in - p_out - p_laplace)
                # amount transfered from outside to inside (= -J)
                Δamount = diff_step * (mu_out - mu_in)

                # check whether the updated droplet vanishes
                volume_vanishes = V + ΔV < volume_min
                fraction_vanishes = np.sum(amounts + Δamount + Sin) < V * phi_min
                if volume_vanishes or fraction_vanishes:
                    # remove droplet & ensure all amount is dumped into the background
                    Δamount = -amounts  # loose all material
                    Sout[:] = 0  # retain the background reaction
                    droplet_data.radius = 0  # remove droplet
                    droplet_data.amounts[...] = 0

                else:
                    # droplet remains -> change droplet volume and composition
                    droplet_data.radius = radius(V + ΔV)  # update volume

                    # limit added material to the space inside the droplet
                    amount_cur_tot = amounts.sum()
                    amount_add_tot = (Δamount + Sin).sum()
                    amount_max = (1 - min_fraction) * V
                    if amount_cur_tot + amount_add_tot > amount_max:
                        # limit transfered amount so that phi_tot does not exceed 1
                        factor = (amount_max - amount_cur_tot) / amount_add_tot
                        amount_corrections += (Δamount + Sin) * (1 - factor)
                        Δamount *= factor
                        Sin *= factor

                    # update compositions of all species
                    # for i in range(num_comps):
                    droplet_data.amounts += Δamount + Sin

                # update the scalar fields at the droplet position and remove chemical
                # reactions that have been run in the background field although this
                # region is occupied by a droplet
                if has_reaction:
                    add_amounts(fields_data, droplet_data.position, -Δamount - Sout)
                else:
                    add_amounts(fields_data, droplet_data.position, -Δamount)

            # update the background field
            fields_data -= dt * j_back
            if has_reaction:
                fields_data += dt * s_out

            with nb.objmode:
                self.diagnostics["amount_corrections"] += amount_corrections

        return evolver  # type: ignore

    def evolve(self, elements: ActorElementType, t: float, dt: float) -> None:  # type: ignore
        """evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple):
                The state of all the droplets and of the field
            t (float):
                The current time point
            dt (float):
                The time step
        """
        self._check_cache(elements)
        droplets_el, fields_el = elements

        # extract constants
        dim = self._cache["dim"]
        num_comps = self._cache["num_comps"]
        volume_relaxation_factor = self.parameters["volume_relaxation_factor"]
        mobility = np.broadcast_to(self.parameters["mobility"], (num_comps,))
        surface_tension = self.parameters["surface_tension"]
        phi_min = self.parameters["dissolve_fraction"]
        volume_min = self._cache["volume_min"]
        min_fraction = self.parameters["min_fraction"]
        has_reaction = self._cache["has_reaction"]

        # get functions
        regularize = self._cache["regularize"]
        calc_state_vars = self._cache["calc_state_vars"]
        reaction_flux = self._cache["reaction_flux"]
        interpolate_fields = self._cache["interpolate_fields"]

        # determine diffusive flux in the background
        bc = self.parameters["boundary_conditions"]
        j_back = [
            -mobility[i] * field.laplace(bc).data  # type: ignore
            for i, field in enumerate(fields_el.field)
        ]

        self.diagnostics.setdefault("amount_corrections", np.zeros(num_comps))

        # determine reaction flux in the background
        if has_reaction:
            phi_back = fields_el.data
            mu_back = (
                np.log(phi_back)
                + self._chis_solvent
                + np.tensordot(self._chis_reduced, phi_back, axes=(1, 0))
            )
            s_out = reaction_flux(phi_back, mu_back, t)

        # update all droplets
        amount_corrections = np.zeros(num_comps)
        for droplet in droplets_el.droplets:
            if droplet.radius <= 0:
                continue  # skip droplets that have disappeared
            V = droplet.volume

            # determine the compositions inside and outside
            phi_out = fields_el.get_concentrations(droplet.position)
            phi_in = droplet.phis + phi_out  # raise above background
            # artificial limit to avoid problems
            amount_corrections += V * (regularize(phi_out) + regularize(phi_in))

            # obtain thermodynamic quantities inside and at the droplet
            _, mu_in, p_in = calc_state_vars(phi_in)
            _, mu_out, p_out = calc_state_vars(phi_out)

            # determine reaction fluxes in the droplet region
            if has_reaction:
                s_in = reaction_flux(phi_in, mu_in, t)
                Sin = dt * V * s_in
                Sout = dt * V * interpolate_fields(s_out, droplet.position)
            else:
                Sin = Sout = 0.0

            # Laplace pressure is exerted onto the droplet
            p_laplace = (dim - 1) * surface_tension / droplet.radius

            # get fluxes as linear functions of the respective forces
            if dim == 1:
                vol_step = dt * mobility.mean() * volume_relaxation_factor
                diff_step = dt * mobility * phi_out
            elif dim == 3:
                factor = dt * 4 * np.pi * droplet.radius * mobility
                vol_step = factor.mean() * volume_relaxation_factor
                diff_step = factor * phi_out
            else:
                raise NotImplementedError("Only implemented for dim ∈ [1, 3]")
            # droplet volume increases as response to pressure difference
            ΔV = vol_step * (p_in - p_out - p_laplace)
            # amount transfered from outside to inside (= -J)
            Δamount = diff_step * (mu_out - mu_in)

            # check whether the updated droplet vanishes
            volume_vanishes = V + ΔV < volume_min
            fraction_vanishes = np.sum(droplet.amounts + Δamount + Sin) < V * phi_min
            if volume_vanishes or fraction_vanishes:
                # remove droplet & ensure all amount is dumped into the background
                Δamount = -droplet.amounts  # loose all material
                Sout = 0
                droplet.radius = 0  # remove droplet
                droplet.amounts = 0

            else:
                # droplet remains -> change droplet volume and composition
                droplet.volume = V + ΔV  # update volume
                # limit added material to the space inside the droplet
                amount_cur_tot = droplet.amounts.sum()
                amount_add_tot = (Δamount + Sin).sum()
                amount_max = (1 - min_fraction) * droplet.volume
                if amount_cur_tot + amount_add_tot > amount_max:
                    # limit transfered amount so that phi_tot does not exceed 1
                    factor = (amount_max - amount_cur_tot) / amount_add_tot
                    amount_corrections += (Δamount + Sin) * (1 - factor)
                    Δamount *= factor
                    Sin *= factor

                # update amounts in droplet and clip it to a permissible range
                droplet.amounts += Δamount + Sin

            # update the scalar fields at the droplet position and remove chemical
            # reactions that have been run in the background field although this region
            # is occupied by a droplet
            fields_el.add_amounts(droplet.position, -Δamount - Sout)

        # update the background field
        for i, field in enumerate(fields_el.field):
            field.data -= dt * j_back[i]
            if has_reaction:
                field.data += dt * s_out[i]

        self.diagnostics["amount_corrections"] += amount_corrections
