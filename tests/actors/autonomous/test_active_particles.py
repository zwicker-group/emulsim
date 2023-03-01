"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest

from sim.actors.autonomous import ActiveParticleActor
from sim.elements import ArrowsElement


def test_active_particles():
    """simple test of active particles"""
    # setup state
    particle_data = np.random.uniform(0, 100, size=(10, 2))
    element = ArrowsElement.from_position_random_direction(particle_data)
    actor = ActiveParticleActor()

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy()
    actor.evolve((element,), 0, 1)
    assert np.all(element.positions != ref.positions)
    np.testing.assert_array_equal(element.directions, ref.directions)

    ref = element.copy()
    evolver = actor.make_evolver_numba((element,))
    evolver((element.data,), 0, 1)
    assert np.all(element.positions != ref.positions)
    np.testing.assert_array_equal(element.directions, ref.directions)


@pytest.mark.parametrize("dim", [1, 2])
def test_active_particles_rotation_diffusion(dim):
    """simple test of Brownian motion of droplets"""
    # setup state
    particle_data = np.random.uniform(0, 100, size=(10, dim))
    element = ArrowsElement.from_position_random_direction(
        particle_data, np.random.uniform(0, 1, 10)
    )
    dir_mag = np.linalg.norm(element.directions)
    actor = ActiveParticleActor({"rotational_diffusion": 1})

    assert actor.estimate_dt((element,)) > 0
    ref = element.copy()
    actor.evolve((element,), 0, 1)
    assert np.all(element.positions != ref.positions)
    assert np.all(element.directions != ref.directions)
    assert np.linalg.norm(element.directions) == pytest.approx(dir_mag)

    ref = element.copy()
    evolver = actor.make_evolver_numba((element,))
    evolver((element.data,), 0, 1)
    assert np.all(element.positions != ref.positions)
    assert np.all(element.directions != ref.directions)
    assert np.linalg.norm(element.directions) == pytest.approx(dir_mag)
