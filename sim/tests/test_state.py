"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.tools.misc import skipUnlessModule

from ..elements.tests.test_generic import generate_elements
from ..state import State


@pytest.mark.parametrize("dim", [1, 2])
def test_state(dim):
    """ test some methods of the SimulationState class """
    s = State({str(i): el for i, el in enumerate(generate_elements(dim))})

    assert isinstance(str(s), str)
    assert isinstance(repr(s), str)
    assert isinstance(s.attributes, dict)
    assert len(s.attributes["elements"]) == len(s)
    assert len(s.data) == len(s)
    assert s.degrees_of_freedom > 0

    s2 = s.copy()
    assert s is not s2
    assert s == s2

    # extract items
    name = list(s.elements.keys())[0]
    assert name in s
    assert s[name] is s.elements[name]

    # test get_quantities
    quantities = s.get_quantity("total_amount", total=False)
    quantity_els = [name for name, element in s if hasattr(element, "total_amount")]
    assert set(quantities.keys()) == set(quantity_els)
    assert s.get_quantity("total_amount", total=True) > 0
    assert len(s.get_quantity("nonsense", total=False)) == 0
    assert s.get_quantity("nonsense", total=True) == 0

    if dim == 2:
        s.plot()


def test_state_errors():
    """ test some safe-guarding of the State class """
    with pytest.raises(ValueError):
        State({str(i): el for i, el in enumerate(generate_elements())})


@skipUnlessModule("h5py")
@pytest.mark.parametrize("dim", [1, 2])
def test_state_io(dim, tmp_path):
    """ test some IO of the State class """
    s1 = State({str(i): el for i, el in enumerate(generate_elements(dim))})

    path = tmp_path / "state.hdf5"
    s1.to_file(path)
    s2 = State.from_file(path)
    assert s1 is not s2
    assert s1 == s2
