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
    """ simple test of box actor """
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
    np.testing.assert_allclose(p1.data, expected.reshape(-1, 1))

    evolver = box.make_evolver_numba((p2,))
    evolver((p2.data,), 1, 1)
    np.testing.assert_allclose(p2.data, expected.reshape(-1, 1))


@pytest.mark.parametrize("periodic", [True, False])
def test_box_actor_reflect_arrows(periodic):
    """ simple test of box actor """
    grid = CartesianGrid([[1, 3]], 1, periodic=periodic)
    box = BoxActor.from_grid(grid)

    coords = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    direction = np.ones(len(coords))
    data = np.c_[coords, direction]
    p1 = ArrowsElement(data)
    p2 = p1.copy()

    if periodic:
        coords_exp = np.array([1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5])
        direction_exp = direction
    else:
        coords_exp = np.array([1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5])
        direction_exp = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
    data_exp = np.c_[coords_exp, direction_exp]

    box.evolve((p1,), 1, 1)
    np.testing.assert_allclose(p1.data, data_exp)

    evolver = box.make_evolver_numba((p2,))
    evolver((p2.data,), 1, 1)
    np.testing.assert_allclose(p2.data, data_exp)
