"""
Provides a coupling of extended spherical droplets to a field

This module also provides a class for managing a collection of spherical shells
with different subdivisions into spherical sectors. Each sector is defined by a
unit vector pointing to its center and an associated weight, which captures is
local size compared to all other shell sectors.

.. autosummary::
   :nosignatures:

   ~ShellSectors
   ~ShellCollection
   ~SphericalDropletActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import warnings
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numba as nb
import numpy as np

from pde import ScalarField
from pde.grids.base import DimensionError
from pde.tools import expressions, spherical
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import FieldElementBase, ReservoirElement, SphericalDropletsElement
from ..base import ActorBase

π = float(np.pi)

import scipy.special as sc

tolerance = float(1e-10)

class ShellSectors:
    """class representing the sectors of a single shell"""

    def __init__(self, vectors: np.ndarray, weights: np.ndarray = None):
        """
        Args:
            vectors (list):
                (Unit) vectors defining the position of the centers of the shell sectors
            weights (list):
                List of weights for each shell sector determining the fraction of the
                droplet surface that is covered by the respective sector. The sum of all
                weights must be one.
        """
        self.vectors = np.asanyarray(vectors)
        if weights is None:
            self.weights = np.full(self.dim, 1 / self.dim)
        else:
            self.weights = np.asanyarray(weights)

        assert len(self.weights) == len(self.vectors)
        assert np.isclose(self.weights.sum(), 1.0)

    @classmethod
    def generate(cls, dim: int, sector_count: int = 1) -> "ShellSectors":
        """generate a :class:`ShellSectors` for a simulation

        Args:
            dim (int):
                The dimension of space
            sector_count (int):
                Number of sectors to generate (ignored when dim==1)

        Note:
            One-dimensional shells are special in that there can only be exactly two
            sectors. Consequently, `sector_count` is not used in this case.

        Returns:
            :class:`ShellSectors`
        """
        if dim == 1:
            # special case where two sectors is the only useful choice
            shell = spherical.PointsOnSphere.make_uniform(dim=1)

        else:  # higher dimensions
            shell = spherical.PointsOnSphere.make_uniform(
                dim=dim, num_points=sector_count
            )
            assert sector_count == len(shell.points)

        weights = shell.get_area_weights(balance_axes=True)
        return cls(shell.points, weights)

    @property
    def dim(self) -> int:
        """int: dimension of the space this shell is defined for"""
        return self.vectors.shape[1]

    @property
    def sector_count(self) -> int:
        """int: number of sectors"""
        return self.vectors.shape[0]

    def get_shell(self, radius: float) -> "ShellSectors":
        """return shell corresponding to droplet of given radius

        Args:
            radius (float):
                The radius of the droplet

        Returns:
            :class:`ShellSectors`: The shell associated with this radius
        """
        return self

    def make_shell_data_getter(
        self,
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
        """returns a function for obtaining a shell

        Returns:
            callable: A function that is called with a radius and returns a
                tuple (numpy.ndarray, numpy.ndarray) of the shell vectors and
                the associated weights. The shell vectors are unit vectors
                pointing from the droplet center to the shell center. The
                weights give the fraction of the droplet surface that is covered
                by the respective shell, so that the sum of all weights is unity
        """
        vectors = self.vectors
        weights = self.weights

        @jit
        def get_shell(radius: float) -> Tuple[np.ndarray, np.ndarray]:
            """compiled helper function that extracts shell parameters"""
            return vectors, weights

        return get_shell  # type: ignore


class ShellCollection:
    """class representing a collection of shells"""

    max_sector_count: int = 512  # maximal number of sectors

    def __init__(
        self,
        shells: Sequence[ShellSectors],
        max_radii: Sequence[float],
        info_dict: Dict[str, Any] = None,
    ):
        """
        Args:
            shells (list):
                List of shells
            max_radii (:class:`~numpy.ndarray`):
                The maximal sphere radius that each shell should be used for
            info_dict (dict, optional):
                A dictionary into which extra information will be stored
        """
        max_radii_ = np.asarray(max_radii, dtype=np.double)

        # order data by max_radii
        idx = np.argsort(max_radii_)
        self.max_radii: np.ndarray = max_radii_[idx]
        self.shells: Sequence[ShellSectors] = [shells[i] for i in idx]

        if len(self.shells) == 0:
            raise RuntimeError("Require at least one shell")

        # self-consistency checks
        assert len(self.shells) == len(self.max_radii)
        assert len(set(s.dim for s in self.shells)) == 1

        self.dim = self.shells[0].dim
        self.usage = [0] * len(self)
        if info_dict is not None:
            info_dict["shell_collection_usage"] = self.usage

    @classmethod
    def from_dictlist(
        cls, dictlist: Sequence[Dict[str, Any]], info_dict: Dict[str, Any] = None
    ) -> "ShellCollection":
        """create shell collection from a list of dictionaries

        Args:
            dictlist (list of dicts):
                a list of shells, where each shell is characterized by a dictionary with
                entries 'vectors', 'weights', and 'radius_threshold'.
            info_dict (dict, optional):
                A dictionary into which extra information will be stored

        Returns:
            :class:`ShellCollection`
        """
        shells, max_radii = [], []
        for d in dictlist:
            shells.append(ShellSectors(d["vectors"], d["weights"]))
            max_radii.append(d["radius_threshold"])
        return cls(shells, max_radii, info_dict=info_dict)

    @classmethod
    def generate(
        cls,
        dim: int,
        sector_size_max: float = 1,
        radius_max: float = np.inf,
        info_dict: Dict[str, Any] = None,
    ) -> "ShellCollection":
        """generate a :class:`ShellCollection` for a simulation

        Args:
            dim (int):
                The dimension of space
            sector_size_max (float):
                Maximal linear size of sectors associated with shell points
            radius_max (float, optional):
                The maximal radius of the sphere that needs to be considered
            info_dict (dict, optional):
                A dictionary into which extra information will be stored

        Note:
            One-dimensional shells are special in that there can only be exactly
            two sectors. Consequently, `max_sector_size` and `radius_max` are
            not used in this case.

        Returns:
            :class:`ShellCollection`
        """
        if dim == 1:
            # special case since only one shell exists
            shell = spherical.PointsOnSphere.make_uniform(dim=1)
            shell_data = {
                "vectors": shell.points,
                "weights": shell.get_area_weights(),
                "radius_threshold": np.inf,
            }
            data = [shell_data]

        else:  # higher dimensions
            # estimate maximal sector area from linear sector size
            sector_area_max = sector_size_max ** (dim - 1)
            sector_count_approx = 2 * dim  # smallest sector count

            # calculate the maximal number of sectors
            if np.isfinite(radius_max):
                # calculate maximal number of sectors necessary
                surface_max = spherical.surface_from_radius(radius_max, dim=dim)
                max_sector_count = int(
                    np.clip(
                        surface_max / sector_area_max,
                        a_min=sector_count_approx,
                        a_max=cls.max_sector_count,
                    )
                )
            else:
                max_sector_count = cls.max_sector_count

            # construct shell vectors of increasing density for various sizes
            data = []
            while sector_count_approx <= max_sector_count:
                sector_count = int(np.floor(sector_count_approx))
                shell = spherical.PointsOnSphere.make_uniform(
                    dim=dim, num_points=sector_count
                )
                assert sector_count == len(shell.points)

                # get maximal radius of a sphere such that the average area for
                # each vertex is equal to `sector_area_max`
                surface_thresh = sector_count * sector_area_max
                radius_thresh = spherical.radius_from_surface(surface_thresh, dim=dim)
                weights = shell.get_area_weights(balance_axes=True)

                shell_data = {
                    "vectors": shell.points,
                    "weights": weights,
                    "radius_threshold": radius_thresh,
                }
                data.append(shell_data)
                sector_count_approx *= np.sqrt(2)

        return cls.from_dictlist(data, info_dict=info_dict)

    def __getitem__(self, index: int) -> ShellSectors:
        """obtain a shell of the collection

        Args:
            index (int):
                The index of the shell

        Returns:
            :class:`ShellSectors`: An object representing the shell
        """
        return self.shells[index]

    def __len__(self) -> int:
        """int: number of shells in this collection"""
        return len(self.shells)

    def __iter__(self):
        """iterate over all shells"""
        for i in range(len(self)):
            yield self[i]

    def get_shell(self, radius: float) -> ShellSectors:
        """return shell corresponding to droplet of given radius

        Args:
            radius (float):
                The radius of the droplet

        Returns:
            :class:`ShellSectors`: The shell associated with this radius
        """
        i: int = np.searchsorted(self.max_radii, radius)  # type: ignore
        if i >= len(self.max_radii):
            warnings.warn(
                "Shell with radius larger than the prepared range was requested"
            )
            i = len(self.max_radii) - 1

        self.usage[i] += 1
        return self[i]

    def make_shell_data_getter(
        self,
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
        """returns a function for obtaining a shell

        Returns:
            callable: A function that is called with a radius and returns a
                tuple (numpy.ndarray, numpy.ndarray) of the shell vectors and
                the associated weights. The shell vectors are unit vectors
                pointing from the droplet center to the shell center. The
                weights give the fraction of the droplet surface that is covered
                by the respective shell, so that the sum of all weights is unity
        """
        max_radii = self.max_radii
        vectors: Tuple[np.ndarray, ...] = tuple(shell.vectors for shell in self.shells)
        weights: Tuple[np.ndarray, ...] = tuple(shell.weights for shell in self.shells)
        num = len(max_radii)

        @jit
        def get_shell(radius: float) -> Tuple[np.ndarray, np.ndarray]:
            """compiled helper function that extracts shell parameters"""
            i = min(np.searchsorted(max_radii, radius), num - 1)  # type: ignore
            return vectors[i], weights[i]

        return get_shell  # type: ignore


ActorElementType = Tuple[SphericalDropletsElement, FieldElementBase]


class SphericalDropletActor(ActorBase):
    """an actor coupling spherical droplets to a field"""

    parameters_default = [
        Parameter(
            "equilibrium_concentration",
            "1e-5 / radius",
            object,
            "Expression for the equilibrium concentration. This expression can contain "
            "the variables `radius`, `position`, and `id` denoting the droplet radius, "
            "its position vector, and its identity (the index in the list of droplets)"
            ", respectively. Alternatively, the value can also be an instance defining "
            "a __call__ method that returns the equilibrium concentration and a "
            "`get_compiled` method that returns a numba compiled function for "
            "calculating it.",
        ),
        Parameter(
            "diffusivity",
            1.0,
            float,
            "Diffusivity in the shell surrounding the droplets",
        ),
        Parameter(
            "reaction_outside",
            "0",
            str,
            "Reaction rate outside the droplet, which determines the production of "
            "droplet material per unit volume in the shell region. This can be an "
            "expression that depends on `position`, the local concentration value `c` "
            "outside the droplet, or the droplets identity `id` (the index in the list "
            "of droplets).",
        ),
        Parameter(
            "reaction_inside",
            "0",
            str,
            "Reaction rate inside the droplet, which determines the production of "
            "droplet material per unit volume. This can be an expression that depends "
            "on the droplet radius `R`, its location `position`, or its identity `id` "
            "(the index in the list of droplets). Use negative values to destroy "
            "droplet material inside the droplet.",
        ),
        Parameter(
            "drift_enabled", True, bool, "Flag determining whether droplets can move"
        ),
        Parameter(
            "shell_thickness",
            "1",
            str,
            "The thickness of the shell around droplets. This can be either a length "
            "in non-dimensional units or an expression that can be parsed with sympy. "
            "In the latter case, the grid discretization is available as the variable "
            "`dx`",
        ),
        Parameter(
            "shell_sector_method",
            "size",
            str,
            "Determines the method that is used to determine the shell sector size. "
            "Possible values are `size` and `count`.",
        ),
        Parameter(
            "shell_sector_size",
            "1",
            str,
            "The typical azimuthal size of a shell sector. This can be either a length "
            "in non-dimensional units or an expression that can be parsed with sympy. "
            "In the latter case, the grid discretization is available as the variable "
            '`dx`. This value is only used when `shell_sector_method == "size"`',
        ),
        Parameter(
            "shell_sector_count",
            6,
            int,
            'The number of shell sectors when `shell_sector_method == "count"`',
        ),
        Parameter(
            "num_threads",
            "1",
            object,
            "The number of threads to use in the parallel update of the droplets. This "
            "can either be a positive integer or `auto`, in which case the number of "
            "threads are based on the value of numba.config.NUMBA_NUM_THREADS.",
        ),
    ]

    element_classes = (SphericalDropletsElement, (ReservoirElement, FieldElementBase))

    def _parse_expressions(self, out: Dict[str, Any] = None) -> Dict[str, Any]:
        """parse expressions that depend on droplet variables

        Args:
            out (dict, optional):
                Dictionary into which the expressions are stored

        Returns:
            A dictionary with the expressions. This is `out` if it was supplied.
            Otherwise, a new dictionary is returned.
        """
        if out is None:
            out = {}

        # define which parameters need to be parsed
        PARAMETER_TRANSLATE_LIST = [
            {
                "from": "equilibrium_concentration",
                "to": "cEqOut",
                "signature": [["position", "pos", "x"], ["radius", "R"], ["i", "id"]],
            },
            {
                "from": "reaction_inside",
                "to": "sBaseIn",
                "signature": [["position", "pos", "x"], ["radius", "R"], ["i", "id"]],
            },
            {
                "from": "reaction_outside",
                "to": "sOut",
                "signature": [["concentration", "phi", "c"], ["i", "id"]],
            },
        ]

        # parse the equilibrium concentration and the reaction rates
        for translate in PARAMETER_TRANSLATE_LIST:

            expr = self.parameters[translate["from"]]  # type: ignore
            if callable(expr):
                # assume that the expression supports the correct syntax
                out[translate["to"]] = expr  # type: ignore
            else:
                # parse the expression
                out[translate["to"]] = expressions.ScalarExpression(  # type: ignore
                    str(expr), translate["signature"], allow_indexed=True  # type: ignore
                )

        return out

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

        # parse the equilibrium concentration and the reaction rates
        self._parse_expressions(self._cache)

        # parse the parameters using initialization values from the background
        discretization = field.grid.typical_discretization
        variables = {"dx": discretization, "discretization": discretization}
        for key in ["shell_thickness", "shell_sector_size"]:
            self._cache[key] = expressions.parse_number(self.parameters[key], variables)

        # get maximal expected radius
        radius_max = min(field._cuboid.size) / 2

        # generate the shell collection
        if self.parameters["shell_sector_method"] == "size":
            sector_size = self._cache["shell_sector_size"]
            shells: Union[ShellCollection, ShellSectors] = ShellCollection.generate(
                self._cache["dim"], sector_size_max=sector_size, radius_max=radius_max
            )
        elif self.parameters["shell_sector_method"] == "count":
            sector_count = self.parameters["shell_sector_count"]
            shells = ShellSectors.generate(droplets.dim, sector_count=sector_count)
        else:
            raise ValueError(
                f"Unknown shell_sector_method: {self.parameters['shell_sector_method']}"
            )
        self._cache["shells"] = shells

    def estimate_dt(self, elements: ActorElementType) -> float:  # type: ignore
        """estimate the maximal time step for simulating this actor

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            float: the maximal time step
        """
        self._check_cache(elements)
        D = float(self.parameters["diffusivity"])
        L = float(self._cache["shell_thickness"])
        return 0.25 * L ** 2 / D

    def get_flux_outside(
        self, radius: float, c_far: float, cEqOut: float, droplet_id: int
    ) -> float:
        """returns the integrated outwards flux at the droplet surface given
        some imposed concentration value at the outer shell

        Note:
            We assume that the flux is integrated over the entire spherical
            surface, so that it needs to be multiplied by the surface fraction
            when only a sector is considered.

        Args:
            radius (float):
                The current droplet radius
            c_far (float):
                The concentration at the outer side of the shell sector
            cEqOut (float):
                The concentration right at the inner side of the shell sector,
                right at the droplet surface.
            droplet_id (int):
                The id of the droplet, i.e., its position in the internal
                droplet list. This is ignored in the standard implementation
                given here, but is required by the interface since it is useful
                in other situations.

        Returns:
            float: the integrated flux in the outward normal direction.
        """
        D = float(self.parameters["diffusivity"])
        L = float(self._cache["shell_thickness"])
        calc_sOut = self._cache["sOut"]

        if self._cache["dim"] == 1:
            # flux for 1d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if ( sOut_cEqOut != 0 or sOut_c_far != 0 ): # Reactions are ON
                if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                    A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                    final_expression = ( 2 * cEqOut * D - 2 * c_far * D - A * L * L) / L

                else: # B is either 0 or a finite value
                    B = ((sOut_c_far - sOut_cEqOut)/(cEqOut - c_far))
                    A = sOut_c_far + B * c_far
                    l = np.sqrt(D / B)
                    final_expression = (-2*D*(-A + B*c_far + (A - B*cEqOut)*np.cosh(L/l))/np.sinh(L/l))/(B*l)

            else: # Reactions are OFF

                final_expression = 2 * D * (cEqOut - c_far) / L

            return final_expression # type: ignore

        elif self._cache["dim"] == 2:
            # flux for 2d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if ( sOut_cEqOut != 0 or sOut_c_far != 0 ): # Reactions are ON
                if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                    A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                    final_expression = A*π*radius*radius + (π*(-4*cEqOut*D + 4*c_far*D + A*L*(L + 2*radius)))/(2*np.log(radius/(L + radius)))

                else: # B is either 0 or a finite value
                    B = ( ( sOut_c_far - sOut_cEqOut ) / ( cEqOut - c_far ) )
                    A = sOut_c_far + B * c_far
                    l = np.sqrt(D / B)
                    final_expression = (2*D*π*((A - B*c_far)*l + (-A + B*cEqOut)*radius*(sc.i1(radius/l)*sc.k0((L + radius)/l) + sc.i0((L + radius)/l)*sc.k1(radius/l))))/(B*l*(sc.i0((L + radius)/l)*sc.k0(radius/l) - sc.i0(radius/l)*sc.k0((L + radius)/l)))

            else: # Reactions are OFF
                term = 2 * D * π * (c_far - cEqOut)
                final_expression = term / (np.log( radius / ( L + radius ) ) )

            return final_expression # type: ignore

        elif self._cache["dim"] == 3:
            # flux for 3d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if ( sOut_cEqOut != 0 or sOut_c_far != 0 ): # Reactions are ON
                if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                    A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                    final_expression = (-2*π*radius*(-6*cEqOut*D*(L + radius) + 6*c_far*D*(L + radius) + A*L*L*(L + 3*radius)))/(3*L)

                else: # B is either 0 or a finite value
                    B = ((sOut_c_far - sOut_cEqOut) / (cEqOut - c_far))
                    A = sOut_c_far + B * c_far
                    l = np.sqrt(D / B)
                    final_expression = (4*D*π*radius*(-((A - B*cEqOut)*(l + radius/np.tanh(L/l))) + (A - B*c_far)*(L + radius)/np.sinh(L/l)))/(B*l)

            else: # Reactions are OFF
                final_expression = (4*(cEqOut - c_far)*D*π*radius*(L + radius))/L

            return final_expression # type: ignore

        else:
            raise NotImplementedError(f"Unsupported dimension: {self._cache['dim']}")

    def _make_flux_outside(self) -> Callable[[float, float, float, int], float]:
        """create a function that calculates the integrated outwards flux at
        the droplet surface given some imposed concentration value at the outer
        shell.

        Returns:
            callable: the function with the signature
                (radius: float, c_far: float, cEqOut: float, droplet_id: int)
                corresponding to :meth:`SphericalDropletActor.get_flux_outside`
        """
        D = float(self.parameters["diffusivity"])
        L = float(self._cache["shell_thickness"])
        sOut = self._cache["sOut"]
        calc_sOut: Callable[[float, int], float] = sOut.get_compiled()

        try:
            no_reaction = sOut.constant and sOut.value == 0
        except AttributeError:
            no_reaction = False  # cannot determine whether reaction is present

        if self._cache["dim"] == 1:
            if no_reaction:
                def flux_outside(R: float, c_far: float, cEqOut: float, droplet_id: int) -> float:
                    """flux for 1d droplet without reaction"""
                    return 2 * D * (cEqOut - c_far) / L

            else:

                def flux_outside(R: float, c_far: float, cEqOut: float, droplet_id: int) -> float:
                    """flux for 1d droplet with reaction"""
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                        final_expression = ( 2 * cEqOut * D - 2 * c_far * D - A * L * L) / L

                    else: # B is either 0 or a finite value
                        B = ((sOut_c_far - sOut_cEqOut)/(cEqOut - c_far))
                        A = sOut_c_far + B * c_far
                        l = np.sqrt(D / B)
                        final_expression = (-2*D*(-A + B*c_far + (A - B*cEqOut)*np.cosh(L/l))/np.sinh(L/l))/(B*l)

                    return final_expression

        elif self._cache["dim"] == 2:
            if no_reaction:

                def flux_outside(R: float, c_far: float, cEqOut: float, droplet_id: int) -> float:
                    """flux for 2d droplet without reaction"""
                    return 2 * π * D * (cEqOut - c_far) / float(np.log1p(L / R))

            else:

                def flux_outside(R: float, c_far: float, cEqOut: float,droplet_id: int) -> float:
                    """ flux for 2d droplet with reaction """
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                        final_expression = A*π*R*R + (π*(-4*cEqOut*D + 4*c_far*D + A*L*(L + 2*R)))/(2*np.log(R/(L + R)))

                    else: # B is either 0 or a finite value
                        B = ( ( sOut_c_far - sOut_cEqOut ) / ( cEqOut - c_far ) )
                        A = sOut_c_far + B * c_far
                        l = np.sqrt(D / B)
                        final_expression = (2*D*π*((A - B*c_far)*l + (-A + B*cEqOut)*R*(sc.i1(R/l)*sc.k0((L + R)/l) + sc.i0((L + R)/l)*sc.k1(R/l))))/(B*l*(sc.i0((L + R)/l)*sc.k0(R/l) - sc.i0(R/l)*sc.k0((L + R)/l)))

                    return final_expression


        elif self._cache["dim"] == 3:
            if no_reaction:

                def flux_outside(R: float, c_far: float, cEqOut: float, droplet_id: int) -> float:
                    """flux for 3d droplet without reaction"""
                    return 4 * π * D * R * (1 + R / L) * (cEqOut - c_far)

            else:

                def flux_outside(R: float, c_far: float, cEqOut: float, droplet_id: int) -> float:
                    """flux for 3d droplet with reaction"""
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if ( (abs(cEqOut - c_far) < tolerance) or (abs(sOut_cEqOut - sOut_c_far) < tolerance) ): # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far. Hence we solve the Reaction_Diffusion eq D∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        A = (sOut_cEqOut + sOut_c_far) / 2 # Approximate the reaction rate at the centre of the shell sector.
                        final_expression = (-2*π*R*(-6*cEqOut*D*(L + R) + 6*c_far*D*(L + R) + A*L*L*(L + 3*R)))/(3*L)

                    else: # B is either 0 or a finite value
                        B = ( (sOut_c_far - sOut_cEqOut) / ( cEqOut - c_far ) )
                        A = sOut_c_far + B * c_far
                        l = np.sqrt(D / B)
                        final_expression = (4*D*π*R*(-((A - B*cEqOut)*(l + R/np.tanh(L/l))) + (A - B*c_far)*(L + R)/np.sinh(L/l)))/(B*l)

                    return final_expression

        else:
            raise NotImplementedError(f"Unsupported dimension: {self._cache['dim']}")

        return flux_outside

    def get_equilibrium_concentrations(
        self, droplets: SphericalDropletsElement
    ) -> np.ndarray:
        """returns the equilibrium concentration outside each droplet

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
            calc_eqout = self._parse_expressions()["cEqOut"]

        # calculate the equilibrium concentration for each droplet
        result = []
        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius > 0:
                result.append(calc_eqout(droplet.position, droplet.radius, droplet_id))

        return np.array(result)

    def plot_shell_points(
        self,
        elements: ActorElementType,
        state_style: Dict[str, Any] = None,
        point_style: Dict[str, Any] = None,
        shell_style: Dict[str, Any] = None,
    ):
        r"""plot all shell points around the droplets of a given state

        Args:
            elements (tuple):
                The state of all the droplets and of the field
            state_style (dict, optional):
                Dictionary with keyword arguments that are used in the
                :meth:`AgentState.plot` call. This affects the style of
                the background and the actual droplets.
            point_style (dict, optional):
                Dictionary with keyword arguments that are used in the
                :meth:`matplotlib.pyplot.plot` call. This affects the style of
                the shell points.
            shell_style (dict, optional):
                Dictionary with keyword arguments that are used in the
                :meth:`matplotlib.patches.Wedge` call that is responsible for
                drawing the shell area.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge

        droplets, field = elements

        if droplets.dim != 2:
            raise NotImplementedError("Can only plot shell points in 2d")

        # parse input and set default styles
        if state_style is None:
            state_style = {}

        if point_style is None:
            point_style = {}
        point_style.setdefault("linestyle", "")
        point_style.setdefault("marker", ".")
        point_style.setdefault("markersize", 4)
        point_style.setdefault("color", "w")

        if shell_style is None:
            shell_style = {}
        shell_style.setdefault("facecolor", "w")
        shell_style.setdefault("edgecolor", "none")
        shell_style.setdefault("alpha", 0.2)

        # plot the background and the droplets
        if isinstance(field, ScalarField):
            self.plot(phase_field=field, **state_style)

        # initialize the shell
        thickness = self._cache["shell_thickness"]

        # plot all shell points for all droplets
        ax = plt.gca()
        for droplet in droplets.droplets:
            shell = self._cache["shells"].get_shell(droplet.radius)
            ring_radius = droplet.radius + thickness
            # plot the shell as an annulus
            annulus = Wedge(
                droplet.position, ring_radius, 0, 360, width=thickness, **shell_style
            )
            ax.add_artist(annulus)
            # plot the shell points on top
            points = droplet.position[None, :] + ring_radius * shell.vectors
            plt.plot(points[:, 0], points[:, 1], **point_style)

    def _make_droplet_evolver_numba(
        self, elements: ActorElementType
    ) -> Callable[[Tuple[np.ndarray], int, float, float, np.ndarray, np.ndarray], None]:
        """create a function to evolve a single droplet from time `t` to `t + dt`

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
        shell_thickness = self._cache["shell_thickness"]
        drift_enabled = bool(self.parameters["drift_enabled"])

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

        dim = self._cache["dim"]
        radius = spherical.make_radius_from_volume_compiled(dim)
        surface = spherical.make_surface_from_radius_compiled(dim)
        volume = spherical.make_volume_from_radius_compiled(dim)

        normalize_point = field.grid.make_normalize_point_compiled()
        get_concentration = field.make_get_concentration_compiled()
        add_amount = field.make_add_amount_compiled()

        calc_flux = jit(self._make_flux_outside())
        get_shell_data = self._cache["shells"].make_shell_data_getter()

        @jit(nogil=True)
        def droplet_update(
            droplet_data: np.recarray,
            droplet_id: int,
            t: float,
            dt: float,
            field_data: np.ndarray,
            field_update: np.ndarray,
        ) -> None:
            """update a single droplet based on the surrounding field"""
            R = droplet_data.radius
            V = volume(R)
            shell_vectors, shell_weights = get_shell_data(R)

            # obtain the material flux across the droplet surface
            cEqIn = cBaseIn
            cEqOut = calc_cEqOut(droplet_data.position, droplet_data.radius, droplet_id)

            # get concentration distribution outside the droplet
            ring_radius = R + shell_thickness
            flux_out = np.empty(len(shell_vectors))
            for i in range(len(shell_vectors)):
                pos = droplet_data.position + ring_radius * shell_vectors[i]
                cShell = get_concentration(field_data, pos)

                # Calculate the integrated fluxes at the droplet surface. The sign
                # of the fluxes is such that positive values indicate outward fluxes
                flux_out[i] = calc_flux(R, cShell, cEqOut, droplet_id)

            # amount taken up from the outside per sector
            amount_per_shell_out = -dt * flux_out * shell_weights
            amount_total_out = amount_per_shell_out.sum()
            # amount produced in the inside
            sBaseIn = calc_sBaseIn(
                droplet_data.position, droplet_data.radius, droplet_id
            )
            amount_total_in = dt * sBaseIn * V

            # update the droplet volume
            dV = (amount_total_in + amount_total_out) / cEqIn
            if V + dV < 0:
                # droplet disappears
                amount_remain = V * cEqIn - amount_total_in
                amount_per_shell_out *= -amount_remain / amount_total_out
                droplet_data.radius = 0.0  # remove all droplet material
            else:
                droplet_data.radius = radius(V + dV)

            # update the scalar field at the droplet surface
            for i in range(len(shell_vectors)):
                pos = droplet_data.position + droplet_data.radius * shell_vectors[i]
                add_amount(field_update, pos, -amount_per_shell_out[i])

            # adjust the droplet position
            if drift_enabled and droplet_data.radius > 0:
                factor = float(dim) / cEqIn / surface(droplet_data.radius)
                for i in range(len(shell_vectors)):
                    for j in range(dim):
                        droplet_data.position[j] += (
                            factor * amount_per_shell_out[i] * shell_vectors[i, j]
                        )
                    normalize_point(droplet_data.position)

        return droplet_update  # type: ignore

    def make_evolver_numba(  # type: ignore
        self, elements: ActorElementType
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], None]:
        """return a function evolve the state from time `t` to `t + dt`

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            callable: A function with signature
                (droplets_data: :class:`~numpy.ndarray`, field_data: :class:`~numpy.ndarray`,
                t: float, dt: float), evolving `droplets_data` and `field_data`
        """
        self._check_cache(elements)
        droplets, field = elements

        # determine the number of threads to use in the simulation
        num_threads = self.parameters["num_threads"]
        if num_threads == "auto":
            num_threads = nb.config.NUMBA_NUM_THREADS
        try:
            num_threads = int(num_threads)
        except TypeError:
            self._logger.warning(
                "Cannot use num_threads == %s. Using a single thread instead.",
                num_threads,
            )
            num_threads = 1  # safe choice

        # make sure there are enough droplets per thread
        num_threads_max = max(1, droplets.droplet_count // 32)
        num_threads = min(num_threads, num_threads_max)
        self._logger.info(
            f"Initialize update routine of %s with %d threads",
            self.__class__.__name__,
            num_threads,
        )

        # obtain function for updating a single droplet
        droplet_update = self._make_droplet_evolver_numba(elements)

        # obtain the signature for the evolver
        droplet_type = nb.typeof(droplets.data)
        field_type = nb.typeof(field.data)

        if num_threads > 1 and isinstance(field.data, np.ndarray):
            # update droplets in chunks on different threads, assuming that the
            # background data is a numpy array
            @jit(
                signature=nb.void(
                    droplet_type,
                    nb.int64,
                    nb.float64,
                    nb.float64,
                    field_type,
                    field_type,
                ),
                nogil=True,
            )
            def evolve_chunk(
                droplets_data: np.ndarray,
                i_start: int,
                t: float,
                dt: float,
                field_data: np.ndarray,
                background_update: np.ndarray,
            ) -> None:
                """evolve a chunk of droplets explicitly"""
                for droplet_id, droplet_data in enumerate(droplets_data, i_start):
                    # skip droplets that have disappeared
                    if droplet_data.radius > 0:
                        droplet_update(
                            droplet_data,
                            droplet_id,
                            t,
                            dt,
                            field_data,
                            background_update,
                        )

            # obtain shape for the temporary array
            data_shape = field.data.shape
            tmp_shape = (num_threads,) + data_shape

            @jit(parallel=True)
            def evolver(
                elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
            ) -> None:
                """evolve all droplets in parallel chunks"""
                droplets_data, field_data = elements_data
                field_update = np.empty(tmp_shape)  # allocate temporary memory
                # calculate size of each chunk
                size = int(np.ceil(len(droplets_data) / num_threads))
                for i in nb.prange(num_threads):
                    # extract a chunk of droplets
                    droplet_list = droplets_data[i * size : (i + 1) * size]
                    # evolve them and collect change in background
                    field_update[i, ...] = 0
                    evolve_chunk(
                        droplet_list, i * size, t, dt, field_data, field_update[i]
                    )
                for i in range(num_threads):
                    field_data += field_update[i]

        else:
            # update all droplets on the same thread
            @jit
            def evolver(
                elements_data: Tuple[np.ndarray, np.ndarray], t: float, dt: float
            ) -> None:
                """evolve all droplets explicitly"""
                droplets_data, field_data = elements_data
                for droplet_id, droplet_data in enumerate(droplets_data):
                    # skip droplets that have disappeared
                    if droplet_data.radius > 0:
                        droplet_update(
                            droplet_data, droplet_id, t, dt, field_data, field_data
                        )

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
        shells = self._cache["shells"]
        calc_sBaseIn = self._cache["sBaseIn"]

        droplets, field = elements

        for droplet_id, droplet in enumerate(droplets.droplets):
            if droplet.radius == 0:
                continue  # skip droplets that have disappeared

            # obtain the material flux across the droplet surface
            cEqIn = droplets.parameters["droplet_concentration"]
            cEqOut = self._cache["cEqOut"](droplet.position, droplet.radius, droplet_id)

            # get the correct shell for this droplet
            shell = shells.get_shell(droplet.radius)

            # get concentration distribution outside the droplet
            shell_radius = droplet.radius + self._cache["shell_thickness"]
            points = droplet.position[None, :] + shell_radius * shell.vectors

            flux_out = np.empty(len(points))

            for i in range(len(points)):

                cShell = field.get_concentration(points[i])
                # Calculate the integrated fluxes at the droplet surface. The sign
                # of the fluxes is such that positive values indicate outward fluxes
                flux_out[i] = self.get_flux_outside(droplet.radius, cShell, cEqOut, droplet_id)

            # amount taken up from the outside per shell
            amount_per_shell_out = -dt * flux_out * shell.weights
            amount_total_out = amount_per_shell_out.sum()
            # amount produced inside the droplet
            sBaseIn = calc_sBaseIn(droplet.position, droplet.radius, droplet_id)
            amount_total_in = dt * sBaseIn * droplet.volume

            # update the droplet volume
            dV = (amount_total_in + amount_total_out) / cEqIn
            if droplet.volume + dV < 0:
                # make sure
                amount_remain = droplet.volume * cEqIn - amount_total_in
                amount_per_shell_out *= -amount_remain / amount_total_out
                droplet.volume = 0  # remove all droplet material
            else:
                droplet.volume = droplet.volume + dV

            # update the scalar field at the droplet boundary
            for i in range(len(shell.vectors)):
                pos = droplet.position + droplet.radius * shell.vectors[i]
                field.add_amount(pos, -amount_per_shell_out[i])

            # adjust the droplet position
            if self.parameters["drift_enabled"] and droplet.radius > 0:
                area = droplet.surface_area
                d = droplets.dim / (cEqIn * area) * amount_per_shell_out @ shell.vectors
                droplet.position = field.grid.normalize_point(droplet.position + d)
