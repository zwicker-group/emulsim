"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import pytest
import numpy as np

from pde.grids import UnitGrid
from pde.grids.base import DimensionError
from droplets import SphericalDroplet, Emulsion

from ..point_droplet import PointDropletActor
from ....elements import MeanfieldElement, SphericalDropletsElement


@pytest.mark.parametrize("dim", [3])
def test_point_droplets(dim):
    """ simple test of point droplets """
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == len(droplets) == 1

    coupling = PointDropletActor()
    assert isinstance(coupling.info, dict)
    assert coupling.num_elements == 2

    assert 0 < coupling.estimate_dt((droplets, field)) < 1000
    total_amount = pytest.approx(droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    radius = pytest.approx(droplets.data[0].radius)

    evolver = coupling.make_evolver_numba((droplets, field))
    droplets.data[0].radius = 1  # reset radius to check whether it agrees
    field.concentration = 0
    evolver((droplets.data, field.data), 0, 0.5)
    assert field.total_amount + droplets.total_amount == total_amount
    assert droplets.total_amount != total_amount
    assert droplets.data[0].radius == radius

    droplets2 = droplets.copy()
    assert droplets2 is not droplets
    assert np.array_equal(droplets2.data, droplets.data)

    # test incompatible dimensions
    droplets = SphericalDropletsElement.from_droplets([SphericalDroplet([1], 1)])
    coupling = PointDropletActor()
    with pytest.raises(DimensionError):
        coupling.make_evolver_numba((droplets, field))


@pytest.mark.parametrize("dim", [3])
def test_point_droplet_coarsening(dim):
    """ simple test of coarsening """
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    emulsion = Emulsion(
        [
            SphericalDroplet(grid.get_random_point(), 0.1),
            SphericalDroplet(grid.get_random_point(), 0.2),
        ]
    )
    droplets = SphericalDropletsElement.from_droplets(emulsion)
    assert droplets.droplet_count == 2

    coupling = PointDropletActor()

    ceq = coupling.get_equilibrium_concentrations(droplets).mean()
    field.concentration = ceq

    total_amount = pytest.approx(field.total_amount + droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.1)
    assert field.total_amount + droplets.total_amount == total_amount

    assert emulsion[0].radius < 0.1
    assert emulsion[1].radius > 0.2
