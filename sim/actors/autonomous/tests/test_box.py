"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest

from pde.grids import CartesianGrid

from ....elements import ArrowsElement, PointsElement
from .. import BoxActor


@pytest.mark.parametrize("periodic", [True, False])
def test_box_actor_reflect(periodic):
    """simple test of box actor"""
    grid = CartesianGrid([[1, 3]], 1, periodic=periodic)
    box = BoxActor.from_grid(grid)

    coords = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    p1 = PointsElement(coords.reshape(-1, 1))
    p2 = p1.copy()

    if periodic:
        expected = np.array([1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5])
    else:
        expected = np.array([1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5])
    box.evolve((p1,), 1, 1)
    np.testing.assert_allclose(p1.positions, expected.reshape(-1, 1))

    evolver = box.make_evolver_numba((p2,))
    evolver((p2.data,), 1, 1)
    np.testing.assert_allclose(p2.positions, expected.reshape(-1, 1))


@pytest.mark.parametrize("periodic", [True, False])
def test_box_actor_reflect_arrows(periodic):
    """simple test of box actor"""
    grid = CartesianGrid([[1, 3]], 1, periodic=periodic)
    box = BoxActor.from_grid(grid)

    coords = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    p1 = ArrowsElement.from_position_direction(coords.reshape(-1, 1), 1)
    assert p1.dim == 1
    p2 = p1.copy()

    if periodic:
        coords_exp = np.array([1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5])
        direction_exp = np.ones_like(coords_exp)
    else:
        coords_exp = np.array([1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5])
        direction_exp = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
    coords_exp = coords_exp.reshape(-1, 1)
    direction_exp = direction_exp.reshape(-1, 1)

    box.evolve((p1,), 1, 1)
    np.testing.assert_allclose(p1.positions, coords_exp)
    np.testing.assert_allclose(p1.directions, direction_exp)

    evolver = box.make_evolver_numba((p2,))
    evolver((p2.data,), 1, 1)
    np.testing.assert_allclose(p2.positions, coords_exp)
    np.testing.assert_allclose(p2.directions, direction_exp)
