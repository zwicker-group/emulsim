"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import Emulsion, SphericalDroplet
from pde import ScalarField, UnitGrid

from sim import DropletElementTracker, Simulation, State
from sim.actors import DiffusionActor, DropletNucleationActor
from sim.elements import MeanfieldElement, ScalarFieldElement, SphericalDropletsElement


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("field_cls", [MeanfieldElement, ScalarFieldElement])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_nucleation(dim, field_cls, backend):
    """Simple test of of nucleation actor."""
    # prepare initial
    if dim == 1:
        grid = UnitGrid([100])
    elif dim == 2:
        grid = UnitGrid([10, 10])
    background_el = field_cls.from_field(ScalarField(grid))
    drop = SphericalDroplet(grid.get_random_point(), radius=0.1)
    drops = Emulsion([], dtype=drop.data.dtype)
    droplets_el = SphericalDropletsElement.from_droplets(drops, maxcount=100)
    # droplets_el = SphericalDropletsElement.from_random(
    #     1, bounds=grid, radius=0.1, maxcount=100
    # )
    state = State({"background": background_el, "droplets": droplets_el})

    # setup simulation
    simulation = Simulation(state)
    if field_cls is ScalarFieldElement:
        simulation.add_actor("background", DiffusionActor())
    nucleation_actor = DropletNucleationActor(
        {
            "prefactor": 1e-3,
            "scale": 1e3,
            "initial_radius": 0.1,
            "randomize_position": dim == 1,
        }
    )
    simulation.add_actor(("droplets", "background"), nucleation_actor)

    # run simulation
    drop_tracker = DropletElementTracker("droplets", 10)
    result = simulation.run(t_range=1e3, tracker=drop_tracker, backend=backend)

    drop_count = [len(e) for e in drop_tracker.emulsions]
    assert np.all(grid.contains_point(drop_tracker.emulsions[-1].data["position"]))
    assert drop_count[0] == 0
    assert result["background"].total_amount < 0
    assert np.all(np.diff(drop_count) >= 0)
    if dim == 1:
        assert 2 < drop_count[-1] < 7
    elif dim == 2:
        assert 7 < drop_count[-1] < 18

    amount_start = state.get_total_quantity("total_amount")
    amount_end = result.get_total_quantity("total_amount")
    assert amount_start == pytest.approx(amount_end)
