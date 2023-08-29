"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np

from droplets import SphericalDroplet
from pde.tools.misc import skipUnlessModule

from sim.actors.autonomous import BrownianMotionActor
from sim.elements import PointsElement, SphericalDropletsElement


def test_brownian_motion_points():
    """simple test of Brownian motion of points"""

    element = PointsElement(np.random.randn(10, 2))
    actor = BrownianMotionActor()

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy(method="data")
    actor.evolve((element,), 0, 1)
    assert np.all(element.data != ref.data)

    ref = element.copy(method="data")
    evolver = actor.make_evolver_numba((element,))
    evolver((element._data_numba,), 0, 1)
    assert np.all(element.data != ref.data)


@skipUnlessModule("droplets")
def test_brownian_motion_droplets():
    """simple test of Brownian motion of droplets"""

    droplets = [SphericalDroplet(np.random.randn(2), 1) for _ in range(10)]
    element = SphericalDropletsElement.from_droplets(droplets)
    actor = BrownianMotionActor()

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy(method="data")
    actor.evolve((element,), 0, 1)
    assert np.all(element.data["position"] != ref.data["position"])
    assert np.all(element.data["radius"] == ref.data["radius"])

    ref = element.copy(method="data")
    evolver = actor.make_evolver_numba((element,))
    evolver((element._data_numba,), 0, 1)
    assert np.all(element.data["position"] != ref.data["position"])
    assert np.all(element.data["radius"] == ref.data["radius"])
