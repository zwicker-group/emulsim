"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import pytest

from pde.grids import UnitGrid

from sim.actors.autonomous import EmittersActor
from sim.elements import MeanfieldElement


def test_emitters():
    """simple test of emitters"""
    grid = UnitGrid([3, 3])
    background = MeanfieldElement(parameters={"bounds": [[0, 3], [0, 3]]})
    assert background.concentration == pytest.approx(0)

    emitters = EmittersActor({"positions": [grid.get_random_point()]})
    assert isinstance(emitters.info, dict)
    assert emitters.num_elements == 1

    assert len(emitters) == 1

    emitters.evolve((background,), 0, 0.5)
    assert background.total_amount == pytest.approx(0.5)

    evolver = emitters.make_evolver_numba((background,))
    evolver((background._data_numba,), 0, 0.5)
    assert background.total_amount == pytest.approx(1)

    emitters2 = emitters.copy()
    assert emitters2 is not emitters
    assert emitters2 == emitters
