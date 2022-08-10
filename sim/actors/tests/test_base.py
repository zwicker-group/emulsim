"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from ...elements import MeanfieldElement, ReservoirElement
from ..base import ActorBase, find_actors


def test_actor_base():
    """test basic functions of the ActorBase class"""
    field = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})
    reservoir = ReservoirElement()

    class TestActor(ActorBase):
        element_classes = (MeanfieldElement,)

    # zero elements
    assert not TestActor.supports_elements(silent=True)
    with pytest.raises(ValueError):
        TestActor.supports_elements()

    # one element
    assert TestActor.supports_elements(MeanfieldElement)
    assert TestActor.supports_elements(field)
    assert not TestActor.supports_elements(ReservoirElement, silent=True)
    assert not TestActor.supports_elements(reservoir, silent=True)
    with pytest.raises(TypeError):
        TestActor.supports_elements(ReservoirElement)

    # two elements
    assert not TestActor.supports_elements(field, reservoir, silent=True)


def test_find_actors():
    """test find_actors function"""
    field = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})

    class TestActor(ActorBase):
        element_classes = (MeanfieldElement,)

    assert TestActor in find_actors(field)
    assert TestActor in find_actors(MeanfieldElement)
    assert TestActor not in find_actors(ReservoirElement)
    assert TestActor in find_actors(MeanfieldElement, unordered=True)
    assert TestActor not in find_actors(ReservoirElement, unordered=True)

    assert len(find_actors(field)) > 1

    class TestAct2(ActorBase):
        element_classes = (MeanfieldElement, ReservoirElement)

    assert TestAct2 in find_actors(MeanfieldElement, ReservoirElement)
    assert TestAct2 not in find_actors(ReservoirElement, MeanfieldElement)
    assert TestAct2 in find_actors(MeanfieldElement, ReservoirElement, unordered=True)
    assert TestAct2 in find_actors(ReservoirElement, MeanfieldElement, unordered=True)
