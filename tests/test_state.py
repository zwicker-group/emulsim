"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from elements.test_generic import generate_elements
from pde.tools.numba import jit

from sim.state import State


@pytest.mark.parametrize("dim", [1, 2])
def test_state_general(dim, capsys):
    """test some methods of the SimulationState class"""
    s = State(
        {str(i): el for i, el in enumerate(generate_elements(dim, incl_obj=False))}
    )

    assert isinstance(str(s), str)
    assert isinstance(repr(s), str)
    assert isinstance(s.attributes, dict)
    assert len(s.data) == len(s)
    assert s.degrees_of_freedom > 0

    s2 = s.copy()
    assert s is not s2
    assert s == s2

    # test basic error estimate
    error_estimator = s._make_error_estimator()
    assert error_estimator(s._data_numba, s2._data_numba) == 0

    # extract items
    names = list(s.elements.keys())
    for i, name in enumerate(names):
        assert name in s
        assert s.get_index(name) == i
        assert s[name] is s.elements[name]
        assert s[i] is s.elements[name]
        assert s[i] is s[-(len(s) - i)]

    size = len(s)
    with pytest.raises(IndexError):
        s[-size - 1]
    with pytest.raises(IndexError):
        s[size]

    assert list(s.keys()) == names
    assert list(s.values()) == list(s.elements.values())
    assert list(s.items()) == list(s)

    # test get_quantities
    quantities = s.get_quantities("total_amount")
    quantity_els = [name for name, element in s if hasattr(element, "total_amount")]
    assert set(quantities.keys()) == set(quantity_els)
    assert s.get_total_quantity("total_amount") > 0
    assert len(s.get_quantities("nonsense")) == 0
    assert s.get_total_quantity("nonsense") == 0

    if dim == 2:
        s.plot()

    @jit
    def printer(state):
        print(state)

    printer(s._data_numba)
    captured = capsys.readouterr()
    assert captured.out != ""


def test_state_errors():
    """test some safe-guarding of the State class"""
    with pytest.raises(ValueError):
        State({str(i): el for i, el in enumerate(generate_elements())})


@pytest.mark.parametrize("dim", [1, 2])
def test_state_io(dim, tmp_path):
    """test some IO of the State class"""
    s1 = State({str(i): el for i, el in enumerate(generate_elements(dim))})

    path = tmp_path / "state.zarr"
    s1.to_file(path)
    s2 = State.from_file(path)
    assert s1 is not s2
    assert s1 == s2
