"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest

from droplets import Emulsion, SphericalDroplet
from pde.grids import UnitGrid
from pde.grids.base import DimensionError

from ....elements import MeanfieldElement, SphericalDropletsElement
from ..point_droplet import PointDropletActor


@pytest.mark.parametrize("dim", [3])
def test_point_droplets_diffusion(dim):
    """simple test of point droplets with diffusive exchange"""
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
def test_point_droplets_diffusion_coarsening(dim):
    """simple test of coarsening with diffusive exchange"""
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


@pytest.mark.parametrize("dim", [1, 2])
def test_point_droplets_linear(dim):
    """simple test of point droplets with linear exchange"""
    grid = UnitGrid([3] * dim)
    field = MeanfieldElement(0, {"bounds": grid.axes_bounds})
    assert field.concentration == pytest.approx(0)

    droplet = SphericalDroplet(grid.get_random_point(), 1)
    droplets = SphericalDropletsElement.from_droplets([droplet])
    assert droplets.droplet_count == len(droplets) == 1

    coupling = PointDropletActor({"flux_model": "linear"})
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


@pytest.mark.parametrize("dim", [1, 2])
def test_point_droplets_linear_coarsening(dim):
    """simple test of coarsening with linear exchange"""
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

    coupling = PointDropletActor({"flux_model": "linear"})

    ceq = coupling.get_equilibrium_concentrations(droplets).mean()
    field.concentration = ceq

    total_amount = pytest.approx(field.total_amount + droplets.total_amount)

    coupling.evolve((droplets, field), 0, 0.1)
    assert field.total_amount + droplets.total_amount == total_amount

    assert emulsion[0].radius < 0.1
    assert emulsion[1].radius > 0.2
