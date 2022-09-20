"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured as s2u

import pde
from pde.tools.numba import jit

from .... import Simulation, State
from ....elements import (
    FieldCollectionElement,
    MulticomponentDroplet,
    MulticomponentDropletsElement,
)
from .. import MulticomponentDropletActor
from ..multicomponent_droplet import _make_regularizer


@pytest.mark.parametrize("do_jit", [True, False])
def test_make_regularizer(do_jit):
    """test regularizer"""
    jitter = jit if do_jit else lambda x: x

    a = np.array([0.48247728, 0.52013119])
    _make_regularizer(2)(a)
    assert sum(a) < 1

    a = np.array([0.48247728, 0.52013119])
    jitter(_make_regularizer(2))(a)
    assert sum(a) < 1

    test_arr = np.array([[-0.1, 0.0, 0.6], [0.5, 1.1, 0.6]])
    res_arr = np.array([[0, 0.0, 0.5], [0.5, 1, 0.5]])

    regularize = jitter(_make_regularizer(2, eps=0))
    phis = test_arr.copy()
    regularize(phis)
    np.testing.assert_allclose(phis, res_arr)

    regularize = jitter(_make_regularizer(2, eps=0))
    for i in range(len(test_arr)):
        phis = test_arr[:, i].copy()
        regularize(phis)
        np.testing.assert_allclose(phis, res_arr[:, i])

    regularize = jitter(_make_regularizer(2, eps=0.1))
    phis = np.array([[-0.1, 0.0, 0.6], [0.5, 1.1, 0.6]])
    regularize(phis)
    np.testing.assert_allclose(phis, np.array([[0.1, 0.1, 0.45], [0.5, 0.8, 0.45]]))


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_multicomponent_droplet_actor(dim, num_comps):
    """test basic multicomponent droplets simulations"""
    # create the background field
    grid = pde.CartesianGrid([[0, 32]] * dim, 1, periodic=True)
    fc = pde.FieldCollection.scalar_random_uniform(num_comps, grid, 0, 0.1 / num_comps)
    bulk = FieldCollectionElement.from_fields(fc)

    # create some droplets
    drop_list = [
        MulticomponentDroplet.from_composition(
            grid.get_random_point(),  # position
            np.random.uniform(1, 2),  # radius
            np.random.uniform(0, 0.9 / num_comps, num_comps),  # composition
        )
        for _ in range(3)
    ]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)

    # create the simulation state
    state = State({"bulk": bulk, "droplets": droplets_element})
    amounts = state.get_total_quantity("amounts")

    # create the dynamics
    simulation = Simulation(state)
    exchange_actor = MulticomponentDropletActor(
        {"chis": np.full(num_comps, 1), "chis_solvent": 3}
    )
    simulation.add_actor(("droplets", "bulk"), exchange_actor)

    res1 = simulation.run(t_range=10, backend="numpy", dt=1e-2, tracker=None)
    res2 = simulation.run(t_range=10, backend="numba", dt=1e-2, tracker=None)

    np.testing.assert_allclose(s2u(res1["droplets"].data), s2u(res2["droplets"].data))
    np.testing.assert_allclose(res1["bulk"].data, res2["bulk"].data)

    # check for material conservation
    np.testing.assert_allclose(res1.get_total_quantity("amounts"), amounts)
    np.testing.assert_allclose(res2.get_total_quantity("amounts"), amounts)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_multicomponent_coexistence(backend):
    """test equilibrium in multicomponent system"""
    grid = pde.CartesianGrid([[0, 32]] * 3, 1, periodic=True)
    fc = pde.FieldCollection.from_scalar_expressions(grid, [0.1])

    bulk = FieldCollectionElement.from_fields(fc)
    drop_list = [MulticomponentDroplet.from_composition([16] * 3, 1, [0.8])]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)
    state = State({"bulk": bulk, "droplets": droplets_element})
    amount = state.get_total_quantity("amounts")[0]

    simulation = Simulation(state)
    chi = 3.0
    exchange_actor = MulticomponentDropletActor({"chis": [[0]], "chis_solvent": chi})
    simulation.add_actor(("droplets", "bulk"), exchange_actor)

    result = simulation.run(t_range=1000, backend=backend, dt=0.1, tracker=None)
    phiOut = result["bulk"].data[0].item()
    phiIn = result["droplets"].droplets[0].phis[0] + phiOut
    assert result.get_total_quantity("amounts")[0] == pytest.approx(amount)
    chiIn_equivalent = np.log(phiIn / (1 - phiIn)) / (2 * phiIn - 1)
    assert chiIn_equivalent == pytest.approx(chi, rel=0.1)
    chiOut_equivalent = np.log(phiOut / (1 - phiOut)) / (2 * phiOut - 1)
    assert chiOut_equivalent == pytest.approx(chi, rel=0.1)
