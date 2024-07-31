"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde.tools.misc import module_available

from sim import Simulation, State
from sim.actors.autonomous import CoalescenceDropletActor
from sim.elements import SphericalDropletsElement


@pytest.mark.skipif(
    not module_available("droplets"), reason="requires `droplets` module"
)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_coalescence(backend):
    """Simple test of droplet coalescence."""

    droplets = [
        SphericalDroplet(np.random.uniform(0, 1, 2), np.random.uniform(1, 2))
        for _ in range(10)
    ]
    element = SphericalDropletsElement.from_droplets(droplets)
    state = State({"droplets": element})

    simulation = Simulation(state)
    simulation.add_actor("droplets", CoalescenceDropletActor())
    assert simulation.estimate_dt() > 0

    result = simulation.run(10, 1, backend=backend)

    assert result["droplets"].droplet_count == 1
    assert result["droplets"].total_amount == pytest.approx(element.total_amount)
