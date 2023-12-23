"""
Provides an actor nucleating droplets from a field

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple, Union

import numpy as np

from droplets.tools import spherical
from pde import CartesianGrid, ScalarField
from pde.grids.base import DimensionError
from pde.tools.numba import jit

from ... import Parameter
from ...elements import FieldElementBase, SphericalDropletsElement
from ..base import ActorBase

ActorElementType = Tuple[SphericalDropletsElement, FieldElementBase]


class DropletNucleationActor(ActorBase):
    r"""actor that nucleates droplets from a field

    The nucleation is based on simple nucleation theory, assuming a nucleation barrier
    that grows linearly with super-saturation :math:`\Delta c`. The nucleation rate
    :math:`k`, defined per unit volume, is assumed to scale exponentially with the
    nucleation barrier,

    .. math::
        k = k_0 \exp(\alpha \Delta c)

    Here, :math:`k_0` is constant pre-factor (determining the nucleation rate at
    vanishing supersaturation) and :math:`\alpha` controls the strength of the influence
    of the supersaturation.
    """

    parameters_default = [
        Parameter(
            "saturation_concentration",
            0,
            float,
            "Saturation concentration above which nucleation can take place. The super-"
            "saturation is defined as the concentration of the field minus this value.",
        ),
        Parameter(
            "initial_radius",
            1.0,
            float,
            "Initial radius of the nucleated droplets. Typically, this should be "
            "chosen a bit larger than the critical radius.",
        ),
        Parameter(
            "prefactor",
            1.0,
            float,
            "Pre-factor :math:`k_0` setting the nucleation rate at vanishing super-"
            "saturation.",
        ),
        Parameter(
            "scale",
            1.0,
            float,
            r"Pre-factor :math:`\alpha` affecting the hight of the nucleation barrier "
            "and thus how strongly the super-saturation affects the nucleation rate.",
        ),
        Parameter(
            "randomize_position",
            False,
            bool,
            "Determines whether the position of a nucleated droplet will be randomized "
            "within each cell. Disabling this feature may accelerate the simulation at "
            "the expense of more regular droplet positioning.",
        ),
    ]

    element_classes = (SphericalDropletsElement, FieldElementBase)

    def _update_cache(self, elements: ActorElementType) -> None:
        """prepare the simulation doing pre-calculations

        Args:
            elements (tuple):
                The state of all the droplets and of the field
        """
        droplets, field = elements

        if field.dim is not None and droplets.dim != field.dim:
            raise DimensionError(
                "Droplets have a different dimension than the background "
                f"({droplets.dim} != {field.dim})"
            )

        self._cache["dim"] = droplets.dim

        grid = field.grid
        if not isinstance(grid, CartesianGrid):
            raise RuntimeError("`DropletNucleationActor` only supports cartesian grids")
        if self.parameters["randomize_position"]:
            dx = grid.discretization
            cells_lower = grid.cell_coords - dx / 2
            cells_upper = grid.cell_coords + dx / 2
            self._cache["cell_bounds"] = np.concatenate(
                [cells_lower[..., np.newaxis, :], cells_upper[..., np.newaxis, :]],
                axis=-2,
            )
        else:
            self._cache["cell_bounds"] = None

    def estimate_dt(self, elements: ActorElementType) -> float:  # type: ignore
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            float: the maximal time step
        """
        self._check_cache(elements)
        _, field = elements

        Δc = max(np.max(field.data) - self.parameters["saturation_concentration"], 0)
        cell_vol = field.grid.cell_volumes.max()
        k = self.parameters["prefactor"] * np.exp(self.parameters["scale"] * Δc)
        return 1.0 / (k * cell_vol)  # type: ignore

    def nucleation_rate(
        self, field: Union[FieldElementBase, ScalarField]
    ) -> ScalarField:
        """return nucleation rate :math:`k` for a given field

        Note that this nucleation rate is actually a nucleation rate density. The rate
        with which a droplet is nucleated anywhere in the system thus given by the
        integral :code:`rate.integral`, if `rate` is the field returned by this function.

        Args:
            field (:class:`FieldElementBase` or :class:`ScalarField`):
                Scalar field element from which material is taken

        Returns:
            float: Estimated number of droplets that are nucleated
        """
        # calculate nucleation rates for each cell in the field grid
        Δc = field.data - self.parameters["saturation_concentration"]
        k = self.parameters["prefactor"] * np.exp(self.parameters["scale"] * Δc)
        return ScalarField(field.grid, k, label="Nucleation rate")

    def estimate_nucleation_count(
        self, field: FieldElementBase, t_range: float
    ) -> float:
        """rough estimate of the number of nucleated droplets

        Args:
            field (:class:`FieldElementBase` or :class:`ScalarField`):
                Scalar field element from which material is taken
            t_range (float):
                Duration of the simulation

        Returns:
            float: Estimated number of droplets that are nucleated
        """
        return t_range * self.nucleation_rate(field).integral  # type: ignore

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
        droplets, field = elements

        dim = self._cache["dim"]
        size = np.prod(field.grid.shape)
        ΔV = field.grid.cell_volumes.reshape(size)
        cell_coords = field.grid.cell_coords.reshape(size, dim)
        volume = spherical.make_volume_from_radius_compiled(dim)
        add_amount = field.make_add_amount_compiled()

        saturation_concentration = self.parameters["saturation_concentration"]
        prefactor = self.parameters["prefactor"]
        scale = self.parameters["scale"]
        initial_radius = self.parameters["initial_radius"]
        cEqIn = droplets.parameters["droplet_concentration"]
        randomize_position = self.parameters["randomize_position"]
        cell_bounds = self._cache["cell_bounds"]
        if randomize_position:
            cell_bounds = cell_bounds.reshape(size, 2, dim)

        no_space_err = (
            "Cannot add droplet, since space in droplet element is exhausted "
            f" ({len(droplets)} slots)."
        )

        @jit
        def evolver(
            elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
        ) -> None:
            """check for nucleation in all grid cells"""
            droplets_data, field_data = elements_data

            # check nucleation for each cell in the field grid
            drop_id = 0
            for idx in range(size):
                # get local nucleation rate
                Δc = field_data.flat[idx] - saturation_concentration
                k = prefactor * ΔV[idx] * np.exp(scale * Δc)

                if np.random.random() < k * dt:
                    # nucleate a new droplet => find empty slot in droplets
                    while True:
                        if droplets_data[drop_id].radius == 0:
                            break  # found empty slot
                        drop_id += 1
                        if drop_id >= len(droplets_data):
                            raise RuntimeError(no_space_err)

                    # add droplet within this cell
                    droplet_data = droplets_data[drop_id]
                    droplet_data.radius = initial_radius
                    if randomize_position:
                        lo, hi = cell_bounds[idx]
                        for i in range(dim):
                            droplet_data.position[i] = np.random.uniform(lo[i], hi[i])
                    else:
                        droplet_data.position[:] = cell_coords[idx]

                    # remove the amount from the scalar field
                    amount_out = cEqIn * volume(droplet_data.radius)
                    add_amount(field_data, droplet_data.position, -amount_out)

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
        droplets, field = elements
        emulsion = droplets.droplets
        cEqIn = droplets.parameters["droplet_concentration"]

        # calculate nucleation rates for each cell in the field grid
        k = self.nucleation_rate(field)
        ΔV = field.grid.cell_volumes
        nucl_rate = ΔV * k.data
        assert nucl_rate.shape == field.grid.shape

        # determine which cells nucleated a droplet
        if nucl_rate.max() > 1 / dt:
            self._logger.warning(
                f"Nucleation rate too large ({nucl_rate.max()}; reduce dt ({dt})"
            )

        # determine grid cells that exhibit nucleation
        nucleation_cells = np.random.random(size=ΔV.shape) < nucl_rate * dt
        positions = field.grid.cell_coords[nucleation_cells]
        if self.parameters["randomize_position"]:
            cell_bounds = self._cache["cell_bounds"][nucleation_cells]

        drop_id = 0
        for i, position in enumerate(positions):
            # find empty slot in droplets
            while True:
                if emulsion[drop_id].radius == 0:
                    break  # found empty slot
                drop_id += 1
                if drop_id >= len(emulsion):
                    raise RuntimeError(
                        "Cannot add droplet, since space in droplet element is "
                        f"exhausted ({len(emulsion)} slots)."
                    )

            # add droplet within this cell
            droplet = emulsion[drop_id]
            droplet.data["radius"] = self.parameters["initial_radius"]
            if self.parameters["randomize_position"]:
                position = np.random.uniform(*cell_bounds[i])
            droplet.data["position"] = position

            # remove the amount from the scalar field
            amount_out = cEqIn * droplet.volume
            field.add_amount(position, -amount_out)
