"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.misc import skipUnlessModule

from .... import Simulation, State
from .. import CoalescenceDropletActor


@skipUnlessModule("droplets")
def test_coalescence():
    """ simple test of droplet coalescence """
    from droplets import SphericalDroplet

    from ....elements import SphericalDropletsElement

    droplets = [SphericalDroplet(np.random.randn(2), 1) for _ in range(10)]
    element = SphericalDropletsElement.from_droplets(droplets)
    state = State({"droplets": element})

    for backend in ["numpy", "numba"]:
        simulation = Simulation(state)
        simulation.add_actor("droplets", CoalescenceDropletActor())
        assert simulation.estimate_dt() > 0

        result = simulation.run(10, 1, backend=backend)

        assert result["droplets"].droplet_count < element.droplet_count
        assert result["droplets"].total_amount == pytest.approx(element.total_amount)
