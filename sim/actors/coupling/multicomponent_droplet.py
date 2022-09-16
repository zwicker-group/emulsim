"""
Provides an actor coupling point-like droplets to a field

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple

import numpy as np

from droplets.tools import spherical
from pde import ScalarField
from pde.fields.base import FieldBase
from pde.grids.base import DimensionError
from pde.tools.expressions import TensorExpression
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import FieldCollectionElement, MulticomponentDropletsElement
from ..base import ActorBase

ActorElementType = Tuple[MulticomponentDropletsElement, FieldCollectionElement]


class SolventFractionError(RuntimeError):
    """error indicating that the solvent fraction was not in [0, 1]"""

    pass


class MulticomponentDropletActor(ActorBase):
    """actor that couples points-like multicomponent droplets to multiple field

    For simplicity, these droplets interact with the field only at one point (their
    position) using a simple linear exchange flux model. This model can be derived in
    the simple case of a Cahn-Hilliard equation with constant mobility, but assumes that
    droplets live in a three-dimensional world.

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
            "mobility",
            1.0,
            object,
            "Mobility outside the droplet. This factor determines the diffusivity of "
            "molecules in the dilute phase and thus how fast droplets change size.",
        ),
        Parameter(
            "surface_tension",
            0.0,
            float,
            "Surface tension determining the Laplace pressure",
        ),
        Parameter(
            "dissolve_amount",
            1e-6,
            float,
            "Minimal total amount in a droplet before it is considered dissolved",
        ),
    ]

    element_classes = (MulticomponentDropletsElement, FieldCollectionElement)

    @property
    def chis_full(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the full interaction matrix including solvent"""
        chis = self.parameters["chis"]
        num_comp = len(chis)
        result = np.zeros((num_comp + 1, num_comp + 1))
        result[:num_comp, :num_comp] = chis
        result[:num_comp, -1] = result[-1, :num_comp] = self.parameters["chis_solvent"]
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
        num_comps = droplets.num_comps
        chis_sol = self._chis_solvent
        chis_red = self._chis_reduced
        assert chis_sol.shape == (num_comps,)
        assert chis_red.shape == (num_comps, num_comps)

        @jit
        def calc_state_vars(
            phis: np.ndarray,
        ) -> Tuple[float, np.ndarray, float]:
            """calculates thermodynamic state variables from composition"""
            phi_sol = 1 - phis.sum()
            if phi_sol < 0:
                raise SolventFractionError("Solvent has negative concentration")

            log_phi_sol = np.log(phi_sol)
            f = phi_sol * log_phi_sol  # entropy of solvent
            mu = np.empty(num_comps)  # chemical potentials
            p = 0  # pressure
            for i in range(num_comps):  # iterate components
                f += phis[i] * np.log(phis[i])  # entropy of component i
                f += chis_sol[i] * phis[i]
                mu[i] = np.log(phis[i]) - log_phi_sol + chis_sol[i]
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
        droplets, fields = elements

        if fields.dim is not None and droplets.dim != fields.dim:
            raise DimensionError(
                "Droplets have a different dimension than the background "
                f"({droplets.dim} != {fields.dim})"
            )
        if droplets.num_comps > fields.num_fields:
            raise RuntimeError(
                "Droplets have more components than there are background fields "
                f"({droplets.num_comps} > {fields.num_fields})"
            )

        self._cache["dim"] = droplets.dim
        self._cache["calc_state_vars"] = self._make_calc_state_vars(droplets)

        # check reactions
        if self.parameters["reactions"] is None:
            # no reactions
            def noop(phis: np.ndarray, mus: np.ndarray) -> np.ndarray:
                return np.zeros_like(phis)

            self._cache["has_reaction"] = False
            self._cache["reaction_flux"] = noop

        elif callable(self.parameters["reactions"]):
            # callable function is given
            self._cache["has_reaction"] = False
            self._cache["reaction_flux"] = jit(self.parameters["reactions"])

        else:
            # assume an expression is given
            expr = TensorExpression(self.parameters["reactions"], ["phi", "mu", "t"])
            self._cache["has_reaction"] = False
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
            data_field: FieldBase = fields.field.copy(label=kind)
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
        droplets, fields = elements

        # obtain constants that need to be used
        num_comps = droplets.num_comps
        dim = self._cache["dim"]
        mobility = self.parameters["mobility"]
        surface_tension = self.parameters["surface_tension"]
        amounts_min = self.parameters["dissolve_amount"]
        has_reaction = self._cache["has_reaction"]

        # obtain functions that need to be used
        calc_state_vars = self._cache["calc_state_vars"]
        radius = spherical.make_radius_from_volume_compiled(self._cache["dim"])
        volume = spherical.make_volume_from_radius_compiled(self._cache["dim"])
        get_concentrations = fields.make_get_concentrations_compiled()
        add_amounts = fields.make_add_amounts_compiled()
        reaction_flux = self._cache["reaction_flux"]

        @jit
        def evolver(
            elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """evolve all droplets explicitly"""
            droplets_data, fields_data = elements_data
            for droplet_data in droplets_data:
                # skip droplets that have disappeared
                if droplet_data.radius <= 0:
                    continue

                # read basic properties of the droplet
                R = droplet_data.radius
                V = volume(R)
                amounts = droplet_data.amounts

                # check whether the droplet has effectively been dissolved
                if amounts.sum() < amounts_min:
                    # remove the droplet completely
                    ΔV = -V

                else:
                    # determine the compositions inside and outside
                    phi_out = get_concentrations(fields_data, droplet_data.position)
                    Δphi_in = amounts / V
                    phi_in = phi_out + Δphi_in

                    # obtain the material flux across the droplet surface
                    _, mu_in, p_in = calc_state_vars(phi_in)
                    _, mu_out, p_out = calc_state_vars(phi_out)

                    # add surface tension effects
                    p_in = (dim - 1) * surface_tension / R

                    # dynamics according to Eq. S4 in SI of Zwicker & Laan, PNAS (2022)
                    rate = dt * 4 * np.pi * droplet_data.radius * mobility
                    ΔV = rate * (p_in - p_out)
                    Δamount = rate * (mu_out - mu_in)

                    if has_reaction:
                        Δamount += V * reaction_flux(phi_in, mu_in, t)

                    for i in range(num_comps):
                        Δamount[i] = max(Δamount[i], -amounts[i])

                # update the droplet volume
                if V + ΔV <= 0:
                    # make sure all amount is dumped into the background phase
                    Δamount = -amounts  # loose all material
                    droplet_data.radius = 0  # remove droplet
                    droplet_data.amounts[...] = 0
                else:
                    # change droplet volume and composition
                    droplet_data.radius = radius(V + ΔV)  # update volume
                    for i in range(num_comps):
                        droplet_data.amounts[i] = amounts[i] + Δamount[i]

                # update the scalar fields at the droplet position
                add_amounts(fields_data, droplet_data.position, -Δamount)

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
        droplets, fields = elements

        # extract constants
        dim = self._cache["dim"]
        mobility = self.parameters["mobility"]
        surface_tension = self.parameters["surface_tension"]
        amounts_min = self.parameters["dissolve_amount"]
        has_reaction = self._cache["has_reaction"]

        # get functions
        calc_state_vars = self._cache["calc_state_vars"]
        reaction_flux = self._cache["reaction_flux"]

        for droplet in droplets.droplets:
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared

            V = droplet.volume

            # check whether the droplet has effectively been dissolved
            if droplet.amounts.sum() < amounts_min:
                # remove the droplet completely
                ΔV = -V

            else:
                # determine the compositions inside and outside
                phi_out = fields.get_concentrations(droplet.position)
                phi_in = droplet.phis + phi_out  # raise above background

                # obtain the material flux across the droplet surface
                _, mu_in, p_in = calc_state_vars(phi_in)
                _, mu_out, p_out = calc_state_vars(phi_out)

                # add surface tension effects
                p_in = (dim - 1) * surface_tension / droplet.radius

                # dynamics according to Eq. S4 in SI of Zwicker & Laan, PNAS (2022)
                rate = dt * 4 * np.pi * droplet.radius * mobility
                ΔV = rate * (p_in - p_out)
                Δamount = rate * (mu_out - mu_in)

                if has_reaction:
                    Δamount += V * reaction_flux(phi_in, mu_in, t)

                # cannot loose more than there is present in the droplets
                np.clip(Δamount, -droplet.amounts, np.inf, out=Δamount)

            # update the droplet volume
            if droplet.volume + ΔV <= 0:
                # make sure all amount is dumped into the background phase
                Δamount = -droplet.amounts  # loose all material
                droplet.volume = 0  # remove droplet
                droplet.amounts = 0
            else:
                # change droplet volume and composition
                droplet.volume = V + ΔV  # update volume
                droplet.amounts += Δamount  # update amounts in droplet

            # update the scalar fields at the droplet position
            fields.add_amounts(droplet.position, -Δamount)

        # check for coalescence
