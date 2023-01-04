"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import numpy as np
import pytest

from sim.actors.function import FunctionActor, NumbaFunctionActor
from sim.elements import MeanfieldElement


def test_function_actor_1el():
    """test function actor with single element"""
    element = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})

    def f_nb(state, t, dt):
        (field_data,) = state
        field_data[:] += dt

    def f_py(state, t, dt):
        (field,) = state
        field.data[:] += dt

    actor_py = FunctionActor(1, f_py)
    actor_nb = NumbaFunctionActor(1, f_nb)

    for actor in [actor_py, actor_nb]:
        element.data[:] = 1
        actor.evolve((element,), 0, 2)
        np.testing.assert_allclose(element.data, np.array([3]))

    with pytest.raises(NotImplementedError):
        actor_py.make_evolver_numba((element,))

    element.data[:] = 1
    evolver = actor_nb.make_evolver_numba((element,))
    evolver((element._data_numba,), 0, 2)
    np.testing.assert_allclose(element.data, np.array([3]))


def test_function_actor_2el():
    """test function actor with two elements"""
    element_a = MeanfieldElement(data=1, parameters={"bounds": [[0, 1]]})
    element_b = MeanfieldElement(data=2, parameters={"bounds": [[0, 1]]})

    def f_py(state, t, dt):
        (field_a, field_b) = state
        field_a.data[:] += dt
        field_b.data[:] -= dt

    def f_nb(state, t, dt):
        (data_a, data_b) = state
        data_a[:] += dt
        data_b[:] -= dt

    actor_py = FunctionActor(2, f_py)
    actor_nb = NumbaFunctionActor(2, f_nb)

    for actor in [actor_py, actor_nb]:
        element_a.data[:] = 1
        element_b.data[:] = 2
        actor.evolve((element_a, element_b), 0, 2)
        np.testing.assert_allclose(element_a.data, np.array([3]))
        np.testing.assert_allclose(element_b.data, np.array([0]))

    with pytest.raises(NotImplementedError):
        actor_py.make_evolver_numba((element_a, element_b))

    element_a.data[:] = 1
    element_b.data[:] = 2
    evolver = actor_nb.make_evolver_numba((element_a, element_b))
    evolver((element_a.data, element_b.data), 0, 2)
    np.testing.assert_allclose(element_a.data, np.array([3]))
    np.testing.assert_allclose(element_b.data, np.array([0]))


@pytest.mark.parametrize("cls", [FunctionActor, NumbaFunctionActor])
def test_function_actor_signature(cls):
    """test signature checking"""
    cls(1, lambda state, t, dt: ...)

    def f(state, t, dt, extra=None):
        pass

    cls(1, f)

    def f1(state, t):
        pass

    with pytest.raises(ValueError):
        cls(1, f1)

    def f2(state, t, dt, extra):
        pass

    with pytest.raises(ValueError):
        FunctionActor(1, f2)
