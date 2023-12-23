"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import SphericalDroplet
from pde import MemoryStorage, ScalarField, UnitGrid

import sim


def test_field_tracker():
    """test the field tracker"""
    # setup state
    field = ScalarField.random_uniform(UnitGrid([32, 32], periodic=True))
    background = sim.ScalarFieldElement.from_field(field)
    state = sim.State({"field": background})

    # setup simulation
    simulation = sim.Simulation(state)
    simulation.add_actor("field", sim.DiffusionActor())

    # run simulation
    storage = MemoryStorage()
    tracker = sim.FieldTracker("field", storage.tracker(interrupts=1))
    result = simulation.run(t_range=3.00001, dt=0.1, backend="numpy", tracker=tracker)

    assert len(storage) == 4
    np.testing.assert_allclose(storage.data[0], field.data)
    for data in storage.data[1:]:
        assert not np.allclose(data, field.data)
    np.testing.assert_allclose(
        storage.data[-1], result["field"].data, rtol=1e-2, atol=1e-2
    )


def test_element_trackers(tmp_path):
    """test DropletElementTracker and Trajectory"""
    # setup state
    grid = UnitGrid([32, 32], periodic=True)
    background = sim.ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(2)]
    droplets = sim.SphericalDropletsElement.from_droplets(droplet_data)
    state = sim.State({"background": background, "droplets": droplets})

    # setup simulation
    simulation = sim.Simulation(state)
    simulation.add_actor("background", sim.DiffusionActor())
    simulation.add_actor(("droplets", "background"), sim.SphericalDropletActor())

    # run simulation
    drop_t = sim.DropletElementTracker("droplets")
    traj_t = sim.TrajectoryTracker(tmp_path / "trajectory")
    simulation.run(t_range=2.5, dt=0.1, backend="numpy", tracker=[drop_t, traj_t])

    # test EmulsionTimeCourse
    np.testing.assert_allclose(drop_t.emulsions.times, [0, 1, 2])
    assert [len(em) for em in drop_t.emulsions] == [2] * 3

    # test DropletTrackList
    assert len(drop_t.droplet_tracks) == 2

    # test Trajectory
    traj = sim.Trajectory(tmp_path / "trajectory")
    np.testing.assert_allclose(traj.times, [0, 1, 2])
    assert traj[0]["droplets"] == droplets
    assert traj[0]["background"] == background
    assert traj[0]["background"].field.fluctuations == pytest.approx(0)
    assert traj[-1]["background"].field.fluctuations > 1e-6  # demand significant change
