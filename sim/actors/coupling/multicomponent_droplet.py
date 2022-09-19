"""
Provides an actor coupling multicomponent droplets to background fields

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Optional, Tuple, Dict, Any

import numba as nb
import numpy as np

from droplets.tools import spherical
from pde import ScalarField
from pde.fields.base import FieldBase
from pde.grids.base import DimensionError
from pde.tools.expressions import TensorExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter
from pde.tools.docstrings import get_text_block

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

    def regularize(phi: np.ndarray) -> Callable[[np.ndarray], float]:
        """regularize a state ensuring variables stay within bounds"""
        if not isinstance(phi, (np.ndarray, nb.types.Array)):
            raise TypeError

        if phi.ndim == 1:
            # a single set of concentrations is given

            def regularize_impl(phi: np.ndarray) -> np.ndarray:
                """regularize a state ensuring variables stay within bounds"""
                correction = np.zeros(num_comps)

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

            def regularize_impl(phi: np.ndarray) -> np.ndarray:
                """regularize a state ensuring variables stay within bounds"""
                correction = np.zeros(num_comps)

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

        return regularize_impl

    if nb.config.DISABLE_JIT:
        # jitting is disabled => return generic python function

        # we here simply supply a 2d array so the more generic implementation
        # is chosen, which works for all cases in the case of numpy
        return regularize(np.empty((num_comps, 2)))

    else:
        # jitting is enabled => return specialized, compiled function
        return nb.generated_jit(nopython=True)(regularize)  # type: ignore

    return regularize


class MulticomponentDropletActor(ActorBase):
    """actor that couples points-like multicomponent droplets to multiple field

    For simplicity, these droplets interact with the field only at one point (their
    position) using a simple linear exchange flux model. This model can be derived in
    the simple case of a Cahn-Hilliard equation with a mobility that scales with the
    fraction. The model assumes that droplets live in a three-dimensional world.

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
            object,
            "Mobility outside the droplet. This factor determines the diffusivity of "
            "molecules in the dilute phase and thus how fast droplets change size. The "
            "corresponding Onsager coefficient is the product of this mobility and the "
            "fraction of the field.",
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
            "dissolve_amount",
            1e-6,
            float,
            "Minimal total amount in a droplet before it is considered dissolved. This "
            "threshold ensures that large droplets that have the same composition as "
            "the background are removed.",
        ),
        Parameter(
            "dissolve_fraction",
            1e-6,
            float,
            "Minimal total fraction in a droplet before it is considered dissolved. "
            "This threshold ensures that large droplets that have the same composition "
            "as the background are removed.",
        ),
    ]

    element_classes = (MulticomponentDropletsElement, FieldCollectionElement)

    @classmethod
    def from_linear_reactions(
        cls,
        parameters: Dict[str, Any],
        rates: np.ndarray,
        production: np.ndarray = None,
    ) -> Optional[
        Callable[[np.ndarray, np.ndarray, float, Optional[np.ndarray]], np.ndarray]
    ]:
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
            raise ValueError("Cannot use parameter `reactions` and explicit reactions.")

        rate_matrix = np.asarray(rates)
        if rate_matrix.ndim == 1:
            rate_matrix = np.diag(rate_matrix)
        num_comps = len(rate_matrix)

        if production is None:
            production_rate = np.zeros(num_comps)
        else:
            production_rate = np.broadcast_to(production, (num_comps,))  # type: ignore

        if np.allclose(rate_matrix, 0) and np.allclose(production_rate, 0):
            parameters["reactions"] = None

        else:

            def droplet_reactions(
                phis: np.ndarray, mus: np.ndarray, t: float, out: np.ndarray = None
            ) -> np.ndarray:
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
        self, droplets: MulticomponentDropletsElement
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray, float]]:
        """create function calculating the state variables

        Args:
            droplets (:class:`MulticomponentDropletsElement`):
                The element describing all the droplets
        """
        num_comps = len(self.parameters["chis"])
        chis_sol = self._chis_solvent
        chis_red = self._chis_reduced
        assert chis_sol.shape == (num_comps,)
        assert chis_red.shape == (num_comps, num_comps)

        @jit
        def calc_state_vars(phis: np.ndarray) -> Tuple[float, np.ndarray, float]:
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
        self._cache["calc_state_vars"] = self._make_calc_state_vars(droplets_el)
        self._cache["interpolate_field"] = fields_el.grid._make_interpolator_compiled()
        self._cache["regularize"] = _make_regularizer(num_comps)

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
    ) -> Tuple[np.ndarray, FieldBase]:
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
        calc_state_vars = self._make_calc_state_vars(droplets)

        # determine data in all droplets
        data_droplets = []
        for droplet in droplets.droplets:
            if droplet.radius > 0:
                phis_out = fields.get_concentrations(droplet.position)
                phis_in = droplet.phis + phis_out  # raise above background
                data_droplets.append(calc_state_vars(phis_in)[data_id])

        # determine data in the background field
        if data_id == 1:
            data_field: FieldBase = fields.fields.copy(label=kind)
        else:
            data_field = ScalarField(fields.grid, label=kind)
        for cell_id in np.ndindex(*fields.grid.shape):
            idx = (...,) + cell_id
            data_field.data[idx] = calc_state_vars(fields.data[idx])[data_id]

        return np.array(data_droplets), data_field

    def make_evolver_numba(  # type: ignore
        self, elements: ActorElementType
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
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
        mobility = self.parameters["mobility"]
        surface_tension = self.parameters["surface_tension"]
        amounts_min = self.parameters["dissolve_amount"]
        phi_min = self.parameters["dissolve_fraction"]
        volume_min = self._cache["volume_min"]
        has_reaction = self._cache["has_reaction"]
        chis_solvent = self._chis_solvent
        chis_reduced = self._chis_reduced

        if dim != 3:
            raise NotImplementedError("Only implemented for dim==3")

        # obtain functions that need to be used
        regularize = self._cache["regularize"]
        interpolate_field = self._cache["interpolate_field"]
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
            elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """evolve all droplets explicitly"""
            droplets_data, fields_data = elements_data

            # determine diffusive flux in the background
            j_back = np.empty_like(fields_data)
            for i in range(num_comps):
                j_back[i] = mobility * laplace(fields_data[i])

            # determine reaction flux in the background
            if has_reaction:
                phi_back = fields_data
                mu_back = np.empty_like(fields_data)
                for i in range(num_comps):
                    mu_back[i] = np.log(phi_back[i]) + chis_solvent[i]
                    for j in range(num_comps):
                        mu_back[i] += chis_reduced[i, j] * phi_back[j]
                s_back = reaction_flux(phi_back, mu_back, t)

            # update all droplets
            amount_corrections = np.zeros(num_comps)
            for droplet_data in droplets_data:
                if droplet_data.radius <= 0:
                    continue  # skip droplets that have disappeared

                # read basic properties of the droplet
                V = volume(droplet_data.radius)
                amounts = droplet_data.amounts

                # determine the compositions inside and outside
                phi_out = interpolate_field(fields_data, droplet_data.position)
                phi_in = amounts / V + phi_out
                amount_corrections += V * (regularize(phi_out) + regularize(phi_in))

                # obtain the material flux across the droplet surface
                _, mu_in, p_in = calc_state_vars(phi_in)
                _, mu_out, p_out = calc_state_vars(phi_out)

                # add surface tension effects
                p_in += (dim - 1) * surface_tension / droplet_data.radius

                # dynamics fluxes as linear functions of the respective forces
                vol_step = dt * 4 * np.pi * droplet_data.radius * mobility
                ΔV = vol_step * (p_in - p_out)
                diff_step = dt * 4 * np.pi * droplet_data.radius * mobility * phi_out
                Δamount = diff_step * (mu_out - mu_in)

                if has_reaction:
                    # determine reaction fluxes inside droplet and in the
                    # corresponding background zone
                    Sin = dt * V * reaction_flux(phi_in, mu_in, t)
                    Sback = dt * V * interpolate_field(s_back, droplet_data.position)
                    # limit the amount of material that can be removed from droplet
                    for i in range(num_comps):
                        Δamount[i] = max(Δamount[i], -amounts[i] - Sin[i])

                else:
                    # there are no reactions
                    Sback = np.zeros(num_comps)
                    # limit the amount of material that can be removed from droplet
                    for i in range(num_comps):
                        Δamount[i] = max(Δamount[i], -amounts[i])

                # check whether the updated droplet vanishes
                volume_vanishes = V + ΔV < volume_min
                amounts_new = amounts + Δamount + Sin
                amount_vanishes = np.sum(amounts_new) < amounts_min
                fraction_vanishes = np.sum(amounts_new) < V * phi_min
                if volume_vanishes or amount_vanishes or fraction_vanishes:
                    # remove droplet & ensure all amount is dumped into the background
                    Δamount = -amounts  # loose all material
                    droplet_data.radius = 0  # remove droplet
                    droplet_data.amounts[...] = 0
                else:
                    # change droplet volume and composition
                    droplet_data.radius = radius(V + ΔV)  # update volume
                    for i in range(num_comps):
                        droplet_data.amounts[i] = amounts_new[i]

                # update the scalar fields at the droplet position and remove chemical
                # reactions that have been run in the background field although this region
                # is occupied by a droplet
                if has_reaction:
                    add_amounts(fields_data, droplet_data.position, -Δamount - Sback)
                else:
                    add_amounts(fields_data, droplet_data.position, -Δamount)

            # update the background field
            fields_data += dt * j_back
            if has_reaction:
                fields_data += dt * s_back

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
        mobility = self.parameters["mobility"]
        surface_tension = self.parameters["surface_tension"]
        amounts_min = self.parameters["dissolve_amount"]
        phi_min = self.parameters["dissolve_fraction"]
        volume_min = self._cache["volume_min"]
        has_reaction = self._cache["has_reaction"]

        if dim != 3:
            raise NotImplementedError("Only implemented for dim==3")

        # get functions
        regularize = self._cache["regularize"]
        calc_state_vars = self._cache["calc_state_vars"]
        reaction_flux = self._cache["reaction_flux"]
        interpolate_field = self._cache["interpolate_field"]

        # determine diffusive flux in the background
        bc = self.parameters["boundary_conditions"]
        j_back = [mobility * field.laplace(bc).data for field in fields_el.fields]

        self.diagnostics.setdefault("amount_corrections", np.zeros(num_comps))

        # determine reaction flux in the background
        if has_reaction:
            phi_back = fields_el.data
            mu_back = (
                np.log(phi_back)
                + self._chis_solvent
                + np.tensordot(self._chis_reduced, phi_back, axes=(1, 0))
            )
            s_back = reaction_flux(phi_back, mu_back, t)

        # update all droplets
        amount_corrections = np.zeros(num_comps)
        for droplet in droplets_el.droplets:
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared
            V = droplet.volume

            # determine the compositions inside and outside
            phi_out = fields_el.get_concentrations(droplet.position)
            phi_in = droplet.phis + phi_out  # raise above background
            # artificial limit to avoid problems
            amount_corrections += V * (regularize(phi_out) + regularize(phi_in))

            # obtain thermodynamic quantities inside and at the droplet
            try:
                _, mu_in, p_in = calc_state_vars(phi_in)
            except SolventFractionError:
                raise
            _, mu_out, p_out = calc_state_vars(phi_out)

            # add Laplace pressure to the internal pressure
            p_in += (dim - 1) * surface_tension / droplet.radius

            # get fluxes as linear functions of the respective forces
            vol_step = dt * 4 * np.pi * droplet.radius * mobility
            ΔV = vol_step * (p_in - p_out)
            diff_step = dt * 4 * np.pi * droplet.radius * mobility * phi_out
            Δamount = diff_step * (mu_out - mu_in)

            # determine reaction fluxes in the droplet region
            if has_reaction:
                Sin = dt * V * reaction_flux(phi_in, mu_in, t)
                Sback = dt * V * interpolate_field(s_back, droplet.position)
            else:
                Sin, Sback = 0.0, 0.0

            # check whether the updated droplet vanishes
            volume_vanishes = V + ΔV < volume_min
            amounts_new = droplet.amounts + Δamount + Sin
            amount_vanishes = np.sum(amounts_new) < amounts_min
            fraction_vanishes = np.sum(amounts_new) < V * phi_min
            if volume_vanishes or amount_vanishes or fraction_vanishes:
                # remove droplet & ensure all amount is dumped into the background
                Δamount = -droplet.amounts  # loose all material
                droplet.radius = 0  # remove droplet
                droplet.amounts = 0

            else:
                # droplet remains -> change droplet volume and composition
                droplet.volume = V + ΔV  # update volume
                # limit added material to the space inside the droplet
                amount_cur_tot = droplet.amounts.sum()
                amount_add_tot = (Δamount + Sin).sum()
                amount_max = (1 - 1e-8) * droplet.volume
                if amount_cur_tot + amount_add_tot > amount_max:
                    # limit transfered amount so that phi_tot does not exceed 1
                    factor = (amount_max - amount_cur_tot) / amount_add_tot
                    amount_corr = (Δamount + Sin) * (1 - factor)
                    self.diagnostics["amount_corrections"] += amount_corr
                    Δamount *= factor
                    Sin *= factor

                # update amounts in droplet and clip it to a permissible range
                droplet.amounts += Δamount + Sin

            # update the scalar fields at the droplet position and remove chemical
            # reactions that have been run in the background field although this region
            # is occupied by a droplet
            fields_el.add_amounts(droplet.position, -Δamount - Sback)

        # update the background field
        for i, field in enumerate(fields_el.fields):
            field.data += dt * j_back[i]
            if has_reaction:
                field.data += dt * s_back[i]

        self.diagnostics["amount_corrections"] += amount_corrections
