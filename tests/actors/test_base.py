"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from sim.actors.base import ActorBase, find_actors
from sim.elements import MeanfieldElement, ReservoirElement


def test_actor_base():
    """test basic functions of the ActorBase class"""
    field = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})
    reservoir = ReservoirElement()

    class BasicActor(ActorBase):
        element_classes = (MeanfieldElement,)

    # zero elements
    assert not BasicActor.supports_elements(silent=True)
    with pytest.raises(ValueError):
        BasicActor.supports_elements()

    # one element
    assert BasicActor.supports_elements(MeanfieldElement)
    assert BasicActor.supports_elements(field)
    assert not BasicActor.supports_elements(ReservoirElement, silent=True)
    assert not BasicActor.supports_elements(reservoir, silent=True)
    with pytest.raises(TypeError):
        BasicActor.supports_elements(ReservoirElement)

    # two elements
    assert not BasicActor.supports_elements(field, reservoir, silent=True)


def test_find_actors():
    """test find_actors function"""
    field = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})

    class TestActor1(ActorBase):
        element_classes = (MeanfieldElement,)

    assert TestActor1 in find_actors(field)
    assert TestActor1 in find_actors(MeanfieldElement)
    assert TestActor1 not in find_actors(ReservoirElement)
    assert TestActor1 in find_actors(MeanfieldElement, unordered=True)
    assert TestActor1 not in find_actors(ReservoirElement, unordered=True)

    assert len(find_actors(field)) > 1

    class TestActor2(ActorBase):
        element_classes = (MeanfieldElement, ReservoirElement)

    assert TestActor2 in find_actors(MeanfieldElement, ReservoirElement)
    assert TestActor2 not in find_actors(ReservoirElement, MeanfieldElement)
    assert TestActor2 in find_actors(MeanfieldElement, ReservoirElement, unordered=True)
    assert TestActor2 in find_actors(ReservoirElement, MeanfieldElement, unordered=True)
