"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured as s2u

import pde

from .... import Simulation, State
from ....elements import (
    FieldCollectionElement,
    MulticomponentDroplet,
    MulticomponentDropletsElement,
)
from .. import MulticomponentDropletActor


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_multicomponent_droplet_actor(dim, num_comps):
    """test basic multicomponent droplets simulations"""
    # create the background field
    grid = pde.CartesianGrid([[0, 32], [0, 32]], 1, periodic=True)
    fc = pde.FieldCollection.scalar_random_uniform(num_comps, grid, 0, 0.1 / num_comps)
    bulk = FieldCollectionElement.from_field(fc)

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

    res1 = simulation.run(t_range=10, backend="numpy", dt=1e-2)
    res2 = simulation.run(t_range=10, backend="numba", dt=1e-2)

    np.testing.assert_allclose(s2u(res1["droplets"].data), s2u(res2["droplets"].data))
    np.testing.assert_allclose(res1["bulk"].data, res2["bulk"].data)

    # check for material conservation
    np.testing.assert_allclose(res1.get_total_quantity("amounts"), amounts)
    np.testing.assert_allclose(res2.get_total_quantity("amounts"), amounts)
