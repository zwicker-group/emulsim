"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np

from pde.tools.misc import skipUnlessModule

from ....elements import PointsElement
from .. import BrownianMotionDropletActor, BrownianMotionPointActor


def test_brownian_motion_points():
    """ simple test of Brownian motion of points """

    element = PointsElement(np.random.randn(10, 2))
    actor = BrownianMotionPointActor()

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy()
    actor.evolve((element,), 0, 1)
    assert np.all(element.data != ref.data)

    ref = element.copy()
    evolver = actor.make_evolver_numba((element,))
    evolver((element.data,), 0, 1)
    assert np.all(element.data != ref.data)


@skipUnlessModule("droplets")
def test_brownian_motion_droplets():
    """ simple test of Brownian motion of droplets """

    from ....elements import SphericalDropletsElement
    from droplets import SphericalDroplet

    droplets = [SphericalDroplet(np.random.randn(2), 1) for _ in range(10)]
    element = SphericalDropletsElement.from_droplets(droplets)
    actor = BrownianMotionDropletActor()

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy()
    actor.evolve((element,), 0, 1)
    assert np.all(element.data["position"] != ref.data["position"])
    assert np.all(element.data["radius"] == ref.data["radius"])

    ref = element.copy()
    evolver = actor.make_evolver_numba((element,))
    evolver((element.data,), 0, 1)
    assert np.all(element.data["position"] != ref.data["position"])
    assert np.all(element.data["radius"] == ref.data["radius"])
