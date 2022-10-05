"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.misc import skipUnlessModule

from .... import Simulation, State
from .. import CoalescenceDropletActor


@skipUnlessModule("droplets")
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_coalescence(backend):
    """simple test of droplet coalescence"""
    from droplets import SphericalDroplet

    from ....elements import SphericalDropletsElement

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
