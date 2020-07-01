'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np

from pde import UnitGrid, ScalarField, MemoryStorage
from droplets import SphericalDroplet

from .. import *



def test_field_tracker():
    """ test the field tracker """
    # setup state
    field = ScalarField.random_uniform(UnitGrid([32, 32], periodic=True))
    background = ScalarFieldElement.from_field(field)
    state = State({'field': background})
    
    # setup simulation
    simulation = Simulation(state)
    simulation.add_actor('field', DiffusionActor())
    
    # run simulation
    storage = MemoryStorage()
    tracker = FieldTracker('field', storage.tracker(interval=1))
    result = simulation.run(t_range=3.00001, dt=0.1, backend='numpy', tracker=tracker)

    assert len(storage) == 4
    np.testing.assert_allclose(storage.data[0], field.data)
    for data in storage.data[1:]:
        assert not np.allclose(data, field.data)
    np.testing.assert_allclose(storage.data[-1], result['field'].data, rtol=1e-2, atol=1e-2)
        
    
    
def test_droplet_element_trackers():
    """ test DropletElementTracker """
    # setup state
    grid = UnitGrid([32, 32], periodic=True)
    background = ScalarFieldElement.from_field(ScalarField(grid, 0.1))
    droplet_data = [SphericalDroplet(grid.get_random_point(), 1) for _ in range(2)]
    droplets = SphericalDropletsElement.from_droplets(droplet_data)
    state = State({'background': background, 'droplets': droplets})
    
    # setup simulation
    simulation = Simulation(state)
    simulation.add_actor('background', DiffusionActor())
    simulation.add_actor(('droplets', 'background'), SphericalDropletActor())
    
    # run simulation
    tracker = DropletElementTracker('droplets', background_grid=grid)
    simulation.run(t_range=2.5, dt=0.1, backend='numpy', tracker=tracker)

    # test EmulsionTimeCourse
    np.testing.assert_allclose(tracker.emulsions.times, [0, 1, 2])
    assert [len(em) for em in tracker.emulsions] == [2] * 3
    assert tracker.emulsions.grid == grid
    
    # test DropletTrackList
    assert len(tracker.droplet_tracks) == 2
    