"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from elements.test_generic import generate_elements

import pde
from pde.tools.numba import jit

from sim.elements import ScalarFieldElement
from sim.state import State


@pytest.mark.parametrize("dim", [1, 2])
def test_state_general(dim, capsys):
    """Test some methods of the SimulationState class."""
    s = State(
        {str(i): el for i, el in enumerate(generate_elements(dim, incl_obj=False))}
    )

    assert isinstance(str(s), str)
    assert isinstance(repr(s), str)
    assert isinstance(s.attributes, dict)
    assert len(s.data) == len(s)
    assert s.degrees_of_freedom > 0

    s2 = s.copy("clean")
    assert s is not s2
    assert s == s2

    # test basic error estimate
    for backend in ["numpy", "numba"]:
        error_estimator = s._make_error_estimator(backend=backend)
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
    """Test some safe-guarding of the State class."""
    with pytest.raises(ValueError):
        State({str(i): el for i, el in enumerate(generate_elements())})


@pytest.mark.parametrize("element", generate_elements())
def test_state_copy(element):
    """Test copying different states."""
    s = State({"el": element})
    s1 = s.copy(method="data")
    s2 = s.copy(method="data")
    assert s1 is not s2 is not s
    assert s1.data is not s2.data is not s.data


def test_field_element_copy():
    """Special tests on field elements, which have special requirements."""
    field = pde.ScalarField.random_normal(pde.UnitGrid([4, 4]))
    e1 = ScalarFieldElement.from_field(field)

    # copy field element directly
    e2 = e1.copy(method="data")
    assert e1 == e2
    assert e1.grid is e2.grid
    assert e1._field is not e2._field
    assert e1.data is not e2.data
    assert e1.field.data is not e2.field.data
    np.testing.assert_array_equal(e1.data, e2.data)

    # copy field element inside state and check whether really only the data is copied
    state = State({"field": e1})
    s_c = state.copy(method="data")
    e2 = s_c["field"]
    assert e1 == e2
    assert e1.grid is e2.grid
    assert e1._field is not e2._field
    assert e1.data is not e2.data
    assert e1.field.data is not e2.field.data
    assert e1.data is e1.field.data
    assert e2.data is e2.field.data
    np.testing.assert_array_equal(e1.data, e2.data)


@pytest.mark.parametrize("dim", [1, 2])
def test_state_io(dim, tmp_path):
    """Test some IO of the State class."""
    s1 = State({str(i): el for i, el in enumerate(generate_elements(dim))})

    path = tmp_path / "state.zarr"
    s1.to_file(path)
    s2 = State.from_file(path)
    assert s1 is not s2
    assert s1 == s2
