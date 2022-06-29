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

import itertools
import warnings
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numba as nb
import numpy as np
import scipy.special as sc

from droplets.tools import spherical
from pde import ScalarField
from pde.grids.base import DimensionError
from pde.tools import expressions
from pde.tools.cache import cached_method
from pde.tools.misc import module_available
from pde.tools.numba import jit
from pde.tools.parameters import Parameter

from ...elements import FieldElementBase, ReservoirElement, SphericalDropletsElement
from ..base import ActorBase

π = float(np.pi)


def points_cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert points from Cartesian to spherical coordinates

    Args:
        points (:class:`~numpy.ndarray`): Points in Cartesian coordinates

    Returns:
        :class:`~numpy.ndarray`: Points (r, θ, φ) in spherical coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3, "Points must have 3 coordinates"

    ps_spherical = np.empty(points.shape)
    # calculate radius in [0, infinity]
    ps_spherical[..., 0] = np.linalg.norm(points, axis=-1)
    # calculate θ in [0, pi]
    ps_spherical[..., 1] = np.arccos(points[..., 2] / ps_spherical[..., 0])
    # calculate φ in [0, 2 * pi]
    ps_spherical[..., 2] = np.arctan2(points[..., 1], points[..., 0]) % (2 * π)
    return ps_spherical


def points_spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """Convert points from spherical to Cartesian coordinates

    Args:
        points (:class:`~numpy.ndarray`):
            Points in spherical coordinates (r, θ, φ)

    Returns:
        :class:`~numpy.ndarray`: Points in Cartesian coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3, "Points must have 3 coordinates"

    sin_θ = np.sin(points[..., 1])
    ps_cartesian = np.empty(points.shape)
    ps_cartesian[..., 0] = points[..., 0] * np.cos(points[..., 2]) * sin_θ
    ps_cartesian[..., 1] = points[..., 0] * np.sin(points[..., 2]) * sin_θ
    ps_cartesian[..., 2] = points[..., 0] * np.cos(points[..., 1])
    return ps_cartesian


def haversine_distance(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Calculate the haversine-based distance between two points on the surface
    of a sphere. Should be more accurate than the arc cosine strategy.
    See, for example: https://en.wikipedia.org/wiki/Haversine_formula

    Adapted from https://github.com/tylerjereddy/spherical-SA-docker-demo
    Licensed under MIT License (see copy in root of this project)

    Args:
        point1 (:class:`~numpy.ndarray`):
            First point(s) on the sphere (given in Cartesian coordinates)
        point2 (:class:`~numpy.ndarray`): Second point on the sphere
            Second point(s) on the sphere (given in Cartesian coordinates)

    Returns:
        :class:`~numpy.ndarray`: The distances between the points
    """
    # note that latitude φ is θ and longitude λ is φ in our notation
    coords = points_cartesian_to_spherical(point1)
    r1, φ1, λ1 = coords[..., 0], coords[..., 1], coords[..., 2]
    coords = points_cartesian_to_spherical(point2)
    r2, φ2, λ2 = coords[..., 0], coords[..., 1], coords[..., 2]

    # check whether both points lie on the same sphere
    assert np.allclose(r1, r2)

    # we rewrite the standard Haversine slightly as long/lat is not the same as
    # spherical coordinates - φ differs by π/4
    factor = (1 - np.cos(λ2 - λ1)) / 2
    arg = (1 - np.cos(φ2 - φ1)) / 2 + np.sin(φ1) * np.sin(φ2) * factor
    return 2 * r1 * np.arcsin(np.sqrt(arg))  # type: ignore


def get_spherical_polygon_area(vertices: np.ndarray, radius: float = 1) -> float:
    """Calculate the surface area of a polygon on the surface of a sphere.
    Based on equation provided here:
    http://mathworld.wolfram.com/LHuiliersTheorem.html
    Decompose into triangles, calculate excess for each

    Adapted from https://github.com/tylerjereddy/spherical-SA-docker-demo
    Licensed under MIT License (see copy in root of this project)

    Args:
        vertices (:class:`~numpy.ndarray`): List of vertices (using Cartesian
            coordinates) that describe the corners of the polygon. The vertices
            need to be oriented.
        radius (float): Radius of the sphere
    """
    # have to convert to unit sphere before applying the formula
    spherical_coordinates = points_cartesian_to_spherical(vertices)
    spherical_coordinates[..., 0] = 1.0
    vertices = points_spherical_to_cartesian(spherical_coordinates)

    n = vertices.shape[0]
    # point we start from
    root_point = vertices[0]
    totalexcess = 0

    # loop from 1 to n-2, with point 2 to n-1 as other vertex of triangle
    # this could definitely be written more nicely
    b_point = vertices[1]
    root_b_dist = haversine_distance(root_point, b_point)
    for i in np.arange(1, n - 1):
        a_point = b_point
        b_point = vertices[i + 1]
        root_a_dist = root_b_dist
        root_b_dist = haversine_distance(root_point, b_point)
        a_b_dist = haversine_distance(a_point, b_point)
        s = (root_a_dist + root_b_dist + a_b_dist) / 2.0
        arg = (
            np.tan(0.5 * s)
            * np.tan(0.5 * (s - root_a_dist))
            * np.tan(0.5 * (s - root_b_dist))
            * np.tan(0.5 * (s - a_b_dist))
        )
        totalexcess += 4 * np.arctan(np.sqrt(arg))
    return totalexcess * radius**2


class PointsOnSphere:
    """class representing points on an n-dimensional unit sphere"""

    def __init__(self, points):
        """
        Args:
            points (:class:`~numpy.ndarray`):
                The list of points on the unit sphere
        """
        self.points = np.asarray(points, dtype=np.double)
        # normalize vectors to force them onto the unit-sphere
        self.points /= np.linalg.norm(self.points, axis=1)[:, np.newaxis]
        self.dim = self.points.shape[-1]

    @classmethod
    def make_uniform(cls, dim: int, num_points: int = None):
        """create uniformly distributed points on a sphere

        Args:
            dim (int): The dimension of space
            num_points (int, optional): The number of points to generate. Note
                that for one-dimensional spheres (intervals), only exactly two
                points can be generated
        """
        if dim == 1:
            # just have two directions in 2d
            if num_points is None:
                num_points = 2
            if num_points != 2:
                raise ValueError("Can only place 2 points in 1d")
            points = [[-1], [1]]

        elif dim == 2:
            if num_points is None:
                num_points = 8
            # distribute points evenly around the circle
            φs = np.linspace(0, 2 * π, num_points, endpoint=False)
            points = np.c_[np.cos(φs), np.sin(φs)]

        elif dim == 3:
            # Distribute points on the unit sphere using a sunflower spiral
            # (inspired by https://stackoverflow.com/a/44164075/932593)
            if num_points is None:
                num_points = 18
            indices = np.arange(0, num_points) + 0.5
            φ = np.arccos(1 - 2 * indices / num_points)
            θ = π * (1 + 5**0.5) * indices

            # convert to Cartesian coordinates
            points = np.c_[np.cos(θ) * np.sin(φ), np.sin(θ) * np.sin(φ), np.cos(φ)]

        elif num_points is None:
            # use vertices of hypercube in n dimensions
            points = [
                p  # type: ignore
                for p in itertools.product([-1, 0, 1], repeat=dim)
                if any(c != 0 for c in p)
            ]

        else:
            raise NotImplementedError()

        # normalize vectors
        return cls(points)

    @cached_method()
    def get_area_weights(self, balance_axes: bool = True):
        """return the weight of each point associated with the unit cell size

        Args:
            balance_axes (bool): Flag determining whether the weights should be
                chosen such that the weighted average of all points is the
                zero vector

        Returns:
            :class:`~numpy.ndarray`: The weight associated with each point
        """
        from scipy import spatial

        points_flat = self.points.reshape(-1, self.dim)
        if self.dim == 1:
            assert points_flat.shape == (2, 1)
            weights = np.array([0.5, 0.5])

        elif self.dim == 2:
            # get angles
            φ = np.arctan2(points_flat[:, 1], points_flat[:, 0])
            idx = np.argsort(φ)
            s0 = φ[idx[0]] + 2 * π - φ[idx[-1]]
            sizes = np.r_[s0, np.diff(φ[idx]), s0]
            weights = (sizes[1:] + sizes[:-1]) / 2
            weights /= 2 * π

        elif self.dim == 3:
            # calculate weights using spherical voronoi construction
            voronoi = spatial.SphericalVoronoi(points_flat)
            voronoi.sort_vertices_of_regions()

            weight_vals = [
                get_spherical_polygon_area(voronoi.vertices[ix])
                for ix in voronoi.regions
            ]
            weights = np.array(weight_vals, dtype=np.double)
            weights /= spherical.surface_from_radius(1, dim=self.dim)

        else:
            raise NotImplementedError()

        if balance_axes:
            weights /= weights.sum()  # normalize weights
            # adjust weights such that all distances are weighted equally, i.e.,
            # the weighted sum of all shell vectors should vanish. Additionally,
            # the sum of all weights needs to be one. To satisfy these
            # constraints simultaneously, the weights are adjusted minimally
            # (in a least square sense).
            matrix = np.c_[points_flat, np.ones(len(points_flat))]
            vector = -weights @ matrix + np.r_[np.zeros(self.dim), 1]
            weights += np.linalg.lstsq(matrix.T, vector, rcond=None)[0]

        return weights.reshape(self.points.shape[:-1])

    def get_distance_matrix(self):
        """calculate the (spherical) distances between each point

        Returns:
            :class:`~numpy.ndarray`: the distance of each point to each other
        """
        from scipy import spatial

        if self.dim == 1:
            raise ValueError("Distances can only be calculated for dim >= 2")

        elif self.dim == 2:
            # use arc length on unit circle to calculate distances
            def metric(a, b):
                # np.clip is necessary to be tolerant toward numerical inaccuracies
                return np.arccos(np.clip(a @ b, -1, 1))

        elif self.dim == 3:
            # calculate distances on sphere using haversine definition
            metric = haversine_distance

        else:
            raise NotImplementedError()

        # determine the distances between all points
        dists = spatial.distance.pdist(self.points, metric)
        return spatial.distance.squareform(dists)

    def get_mean_separation(self) -> float:
        """float: calculates the mean distance to the nearest neighbor"""
        if len(self.points) < 1:
            return float("nan")

        dists_sorted = np.sort(self.get_distance_matrix(), axis=1)
        return float(dists_sorted[:, 1].mean())

    def write_to_xyz(self, path: str, comment: str = "", symbol: str = "S"):
        """write the point coordinates to a xyz file

        Args:
            path (str): location of the file where data is written
            comment (str, optional): comment that is written to the second line
            symbol (str, optional): denotes the symbol used for the atoms
        """
        with open(path, "w") as fp:
            fp.write("%d\n" % len(self.points))
            fp.write(comment + "\n")
            for point in self.points:
                point_str = " ".join(("%.12g" % v for v in point))
                line = "%s %s\n" % (symbol, point_str)
                fp.write(line)


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
            shell = PointsOnSphere.make_uniform(dim=1)

        else:  # higher dimensions
            shell = PointsOnSphere.make_uniform(dim=dim, num_points=sector_count)
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
            shell = PointsOnSphere.make_uniform(dim=1)
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
                shell = PointsOnSphere.make_uniform(dim=dim, num_points=sector_count)
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
            i = int(min(np.searchsorted(max_radii, radius), num - 1))  # type: ignore
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
            "the variables `position`, `radius`, and `id` denoting the droplet radius, "
            "its position vector, and its identity (the index in the list of droplets)"
            ", respectively. Alternatively, the value can also be an instance defining "
            "a __call__ method that returns the equilibrium concentration and a "
            "`get_compiled` method that returns a numba compiled function for "
            "calculating it. These functions must have the signature "
            "(position, radius, i).",
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
            "mean_reaction_inside",
            "automatic",
            str,
            "Mean reaction rate inside the droplet, which determines the production of "
            "droplet material per unit volume. This can be an expression that depends "
            "on the droplet radius `R`, its location `position`, or its identity `id` "
            "(the index in the list of droplets). Use negative values to destroy "
            "droplet material inside the droplet. The special value `automatic` will "
            "determine this rate by using the parameter `reaction_outside` evaluated "
            "at `droplet_concentration` specified by the droplets.",
        ),
        Parameter(
            "drift_enabled", True, bool, "Flag determining whether droplets can move"
        ),
        Parameter(
            "shell_thickness",
            "dx",
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
            "dx",
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

    reaction_rate_tolerance: float = 1e-10
    """float: tolerance determining when a simpler expression for reactions is used"""

    def _parse_expression(self, parameter_key: str) -> Callable:
        """parse expressions that depend on droplet variables

        Args:
            out (dict, optional):
                Dictionary into which the expressions are stored

        Returns:
            A dictionary with the expressions. This is `out` if it was supplied.
            Otherwise, a new dictionary is returned.
        """
        # define common signatures
        SIGNATURE = {
            "equilibrium_concentration": [
                ["position", "pos", "x"],
                ["radius", "R"],
                ["i", "id"],
            ],
            "mean_reaction_inside": [
                ["position", "pos", "x"],
                ["radius", "R"],
                ["i", "id"],
            ],
            "reaction_outside": [["concentration", "phi", "c"], ["i", "id"]],
        }
        # obtain the parameter and parse further
        expr = self.parameters[parameter_key]
        if callable(expr):
            # assume that the expression supports the correct syntax
            return expr  # type: ignore
        else:
            # parse the expression
            return expressions.ScalarExpression(
                str(expr), SIGNATURE.get(parameter_key), allow_indexed=True
            )

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

        # parse the equilibrium concentration and the reaction rates outside
        self._cache["cEqOut"] = self._parse_expression("equilibrium_concentration")
        self._cache["sOut"] = self._parse_expression("reaction_outside")

        # determine the reaction rate inside droplets
        if self.parameters["mean_reaction_inside"] == "automatic":
            self._logger.info("Determine reaction rate inside droplets automatically")
            cEqIn = droplets.parameters["droplet_concentration"]
            sBaseIn = self._cache["sOut"](cEqIn, 0)
            self.parameters["mean_reaction_inside"] = str(sBaseIn)
        self._cache["sBaseIn"] = self._parse_expression("mean_reaction_inside")

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

        The time step is based on the time scale of diffusion in the shell. In the
        special case of large shells (for instance in mean-field coarsening simulations)
        the defining length scale is the mean droplet radius instead. This estimate is
        based on the growth rate of droplets, dR/dt ∝ D/R, where D is the diffusivity
        and R is the mean droplet radius. Therefore, ΔR/Δt ∝ D/R and ΔR/R = ε implies
        Δt = ε * R**2 / D, where we chose ε=0.1.

        Args:
            elements (tuple):
                The state of all the droplets and of the field

        Returns:
            float: the maximal time step
        """
        self._check_cache(elements)
        droplets, field = elements

        # determine minimal dt based on diffusion in shell
        shell_thickness = float(self._cache["shell_thickness"])
        if droplets.droplet_count > 0:
            mean_radius = float(droplets.data["radius"].mean())
            length_scale = min(shell_thickness, mean_radius)
        else:
            length_scale = shell_thickness

        # ensure that characteristic length scale is not too small
        grid_size = max(bounds[1] - bounds[0] for bounds in field.grid.axes_bounds)
        length_scale = max(length_scale, 1e-4 * grid_size)

        # calculate time scale from length scale and diffusivity
        return 0.1 * length_scale**2 / float(self.parameters["diffusivity"])

    def get_flux_outside(
        self, radius: float, c_far: float, cEqOut: float, droplet_id: int
    ) -> float:
        """returns the integrated outwards flux at the droplet surface given
        some imposed concentration value at the outer shell

        Note:
            We assume that the flux is integrated over the entire spherical
            surface, so that it needs to be multiplied by the surface fraction
            when only a sector is considered. The fluxes are calcuated by solving
            the ReactionDiffusion or the Diffusion equation inside each shell sector.
            Detailed documentation for calculating the material fluxes
            (both inside and outside the droplets) is located at
            /py-sim/docs/methods.

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
        TOLERANCE = 1e-10

        D = float(self.parameters["diffusivity"])
        L = float(self._cache["shell_thickness"])
        calc_sOut = self._cache["sOut"]

        if self._cache["dim"] == 1:
            # flux for 1d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if sOut_cEqOut != 0 or sOut_c_far != 0:  # Reactions are ON
                if (abs(cEqOut - c_far) < TOLERANCE) or (
                    abs(sOut_cEqOut - sOut_c_far) < TOLERANCE
                ):
                    # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far.
                    # As k->0, we solve
                    # the Reaction_Diffusion eq D ∇^2(phi) + A = 0, where
                    # A = (sOut_cEqOut + sOut_c_far) / 2

                    # Approximate the reaction rate at the center of the shell sector.
                    A = (sOut_cEqOut + sOut_c_far) / 2
                    final_expression = (2 * (cEqOut - c_far) * D) / L - A * L

                else:  # k is a finite value
                    k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                    A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (-cEqOut + c_far)
                    ξ = np.sqrt(D / k)  # Reaction-Diffusion length-scale
                    term = -A + k * c_far + (A - k * cEqOut) * np.cosh(L / ξ)
                    final_expression = (-2 * D * term / np.sinh(L / ξ)) / (k * ξ)

            else:  # Reactions are OFF

                final_expression = 2 * D * (cEqOut - c_far) / L

            return final_expression  # type: ignore

        elif self._cache["dim"] == 2:
            # flux for 2d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if sOut_cEqOut != 0 or sOut_c_far != 0:  # Reactions are ON
                if (abs(cEqOut - c_far) < TOLERANCE) or (
                    abs(sOut_cEqOut - sOut_c_far) < TOLERANCE
                ):
                    # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far.
                    # As k->0, we solve
                    # the Reaction_Diffusion eq D ∇^2(phi) + A = 0, where
                    # A = (sOut_cEqOut + sOut_c_far) / 2

                    # Approximate the reaction rate at the center of the shell sector.
                    A = (sOut_cEqOut + sOut_c_far) / 2
                    term = π * (
                        -4 * cEqOut * D + 4 * c_far * D + A * L * (L + 2 * radius)
                    )
                    final_expression = A * π * radius * radius + term / (
                        2 * np.log(radius / (L + radius))
                    )

                else:  # k is a finite value
                    k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                    A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (-cEqOut + c_far)
                    ξ = np.sqrt(D / k)  # Reaction-Diffusion length-scale
                    r1, r2 = radius / ξ, (L + radius) / ξ
                    term1 = (A - k * c_far) * ξ - (A - k * cEqOut) * radius * (
                        sc.i1(r1) * sc.k0(r2) + sc.i0(r2) * sc.k1(r1)
                    )
                    term2 = sc.i0(r2) * sc.k0(r1) - sc.i0(r1) * sc.k0(r2)
                    final_expression = (2 * D * π * term1) / (k * ξ * term2)

            else:  # Reactions are OFF
                term = 2 * D * π * (c_far - cEqOut)
                final_expression = term / (np.log(radius / (L + radius)))

            return final_expression  # type: ignore

        elif self._cache["dim"] == 3:
            # flux for 3d droplet
            sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
            sOut_c_far = calc_sOut(c_far, droplet_id)

            if sOut_cEqOut != 0 or sOut_c_far != 0:  # Reactions are ON
                if (abs(cEqOut - c_far) < TOLERANCE) or (
                    abs(sOut_cEqOut - sOut_c_far) < TOLERANCE
                ):
                    # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far.
                    # As k->0, we solve
                    # the Reaction_Diffusion eq D ∇^2(phi) + A = 0, where
                    # A = (sOut_cEqOut + sOut_c_far) / 2

                    # Approximate the reaction rate at the center of the shell sector.
                    A = (sOut_cEqOut + sOut_c_far) / 2
                    term = (
                        -6 * cEqOut * D * (L + radius)
                        + 6 * c_far * D * (L + radius)
                        + A * L * L * (L + 3 * radius)
                    )
                    final_expression = (-2 * π * radius * term) / (3 * L)

                else:  # k is a finite value
                    k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                    A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (-cEqOut + c_far)
                    ξ = np.sqrt(D / k)
                    term = -((A - k * cEqOut) * (ξ + radius / np.tanh(L / ξ))) + (
                        A - k * c_far
                    ) * (L + radius) / np.sinh(L / ξ)
                    final_expression = (4 * D * π * radius * term) / (k * ξ)

            else:  # Reactions are OFF
                final_expression = (
                    4 * (cEqOut - c_far) * D * π * radius * (L + radius)
                ) / L

            return final_expression  # type: ignore

        else:
            raise NotImplementedError(f"Unsupported dimension: {self._cache['dim']}")

    def _make_flux_outside(self) -> Callable[[float, float, float, int], float]:
        """create a function that calculates the integrated outwards flux at
        the droplet surface given some imposed concentration value at the outer
        shell. The fluxes are calcuated by solving
        the ReactionDiffusion or the Diffusion equation inside each shell sector.
        Detailed documentation for calculating the material fluxes
        (both inside and outside the droplets) is located at
        /py-sim/docs/methods.

        Returns:
            callable: the function with the signature
                (radius: float, c_far: float, cEqOut: float, droplet_id: int)
                corresponding to :meth:`SphericalDropletActor.get_flux_outside`
        """
        tolerance = self.reaction_rate_tolerance
        D = float(self.parameters["diffusivity"])
        L = float(self._cache["shell_thickness"])
        sOut = self._cache["sOut"]
        calc_sOut: Callable[[float, int], float] = sOut.get_compiled()

        try:
            no_reaction = sOut.constant and sOut.value == 0
        except AttributeError:
            no_reaction = False  # reaction seems to be present

        if self._cache["dim"] == 1:
            if no_reaction:

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 1d droplet without reaction"""
                    return 2 * D * (cEqOut - c_far) / L

            else:

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 1d droplet with reaction"""
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if (abs(cEqOut - c_far) < tolerance) or (
                        abs(sOut_cEqOut - sOut_c_far) < tolerance
                    ):
                        # parameters are in a limiting case that we treat separately

                        # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far, so k is not
                        # well defined. However, reactions are weak, so we solve
                        # D ∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        # is the reaction rate at the center of the shell sector
                        A = (sOut_cEqOut + sOut_c_far) / 2
                        final_expression = (2 * (cEqOut - c_far) * D) / L - A * L

                    else:  # k is a finite value
                        k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                        A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (
                            c_far - cEqOut
                        )
                        ξ = np.sqrt(D / k)  # Reaction-diffusion length-scale
                        term = -A + k * c_far + (A - k * cEqOut) * np.cosh(L / ξ)
                        final_expression = (-2 * D * term / np.sinh(L / ξ)) / (k * ξ)

                    return final_expression

        elif self._cache["dim"] == 2:
            if no_reaction:

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 2d droplet without reaction"""
                    return 2 * π * D * (cEqOut - c_far) / float(np.log1p(L / R))

            else:

                if not module_available("numba_scipy"):
                    self._logger.error(
                        "Python package `numba_scipy` is not installed. This package "
                        "is required to compile the Bessel function appearing in 2D."
                    )

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 2d droplet with reaction"""
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if (abs(cEqOut - c_far) < tolerance) or (
                        abs(sOut_cEqOut - sOut_c_far) < tolerance
                    ):
                        # parameters are in a limiting case that we treat separately

                        # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far, so k is not
                        # well defined. However, reactions are weak, so we solve
                        # D ∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        # is the reaction rate at the center of the shell sector
                        A = (sOut_cEqOut + sOut_c_far) / 2
                        term = π * (
                            -4 * cEqOut * D + 4 * c_far * D + A * L * (L + 2 * R)
                        )
                        final_expression = A * π * R * R + term / (
                            2 * np.log(R / (L + R))
                        )

                    else:  # k is a finite value
                        k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                        A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (
                            -cEqOut + c_far
                        )
                        ξ = np.sqrt(D / k)  # Reaction-Diffusion length-scale
                        r1, r2 = R / ξ, (L + R) / ξ
                        term1 = (A - k * c_far) * ξ - (A - k * cEqOut) * R * (
                            sc.i1(r1) * sc.k0(r2) + sc.i0(r2) * sc.k1(r1)
                        )
                        term2 = sc.i0(r2) * sc.k0(r1) - sc.i0(r1) * sc.k0(r2)
                        final_expression = (2 * D * π * term1) / (k * ξ * term2)

                    return final_expression  # type: ignore

        elif self._cache["dim"] == 3:
            if no_reaction:

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 3d droplet without reaction"""
                    return 4 * π * D * R * (1 + R / L) * (cEqOut - c_far)

            else:

                def flux_outside(
                    R: float, c_far: float, cEqOut: float, droplet_id: int
                ) -> float:
                    """flux for 3d droplet with reaction"""
                    sOut_cEqOut = calc_sOut(cEqOut, droplet_id)
                    sOut_c_far = calc_sOut(c_far, droplet_id)

                    if (abs(cEqOut - c_far) < tolerance) or (
                        abs(sOut_cEqOut - sOut_c_far) < tolerance
                    ):
                        # parameters are in a limiting case that we treat separately

                        # If cEqOut ~ c_far, then sOut_cEqOut ~ sOut_c_far, so k is not
                        # well defined. However, reactions are weak, so we solve
                        # D ∇^2(phi) + A = 0, where A = (sOut_cEqOut + sOut_c_far) / 2
                        # is the reaction rate at the center of the shell sector
                        A = (sOut_cEqOut + sOut_c_far) / 2
                        term = (
                            -6 * cEqOut * D * (L + R)
                            + 6 * c_far * D * (L + R)
                            + A * L * L * (L + 3 * R)
                        )
                        final_expression = (-2 * π * R * term) / (3 * L)

                    else:  # k is a finite value
                        k = (sOut_c_far - sOut_cEqOut) / (cEqOut - c_far)
                        A = (c_far * sOut_cEqOut - cEqOut * sOut_c_far) / (
                            -cEqOut + c_far
                        )
                        ξ = np.sqrt(D / k)
                        term1 = -(A - k * cEqOut) * (ξ + R / np.tanh(L / ξ))
                        term2 = (A - k * c_far) * (L + R) / np.sinh(L / ξ)
                        final_expression = (4 * D * π * R * (term1 + term2)) / (k * ξ)

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
            calc_eqout = self._parse_expression("equilibrium_concentration")
            self._cache["cEqOut"] = calc_eqout  # cache for next use

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
                flux_out[i] = self.get_flux_outside(
                    droplet.radius, cShell, cEqOut, droplet_id
                )

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

                # Note: Currently amount_total_in has no influence on the droplet position
                # as amount_total_in is isotropic with respect to azimuthal and polar angle.
                # Hence, amount_total_in contributes only in droplet growth.

                area = droplet.surface_area
                d = droplets.dim / (cEqIn * area) * amount_per_shell_out @ shell.vectors
                droplet.position = field.grid.normalize_point(droplet.position + d)
