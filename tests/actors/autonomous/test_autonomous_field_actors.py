"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, CartesianGrid, DiffusionPDE, FieldCollection, ScalarField, UnitGrid
from pde.tools.misc import module_available

from emulsim.actors.autonomous.fields import (
    CollectionPDEActor,
    DiffusionActor,
    LocalReactionsActor,
    ReactionDiffusionActor,
    ScalarPDEActor,
)
from emulsim.elements import (
    FieldCollectionElement,
    MeanfieldElement,
    ScalarFieldElement,
)


def test_diffusion_actor():
    """Test basic methods of the simple diffusion actor."""
    grid = CartesianGrid([[0, 10]], 5, periodic=True)
    element = ScalarFieldElement.from_field(ScalarField(grid, 3))
    assert element.grid == grid
    actor = DiffusionActor()
    assert isinstance(actor.estimate_dt((element,)), float)
    assert actor.num_elements == 1


def test_diffusion_vs_pde(rng):
    """Compare the diffusion background with the PDF actor."""
    field = ScalarField.random_uniform(UnitGrid([10]), rng=rng)
    e1 = ScalarFieldElement.from_field(field)
    e2 = e1.copy(method="data")
    e3 = e1.copy(method="data")

    a1 = DiffusionActor()
    a2 = ScalarPDEActor(DiffusionPDE())
    assert isinstance(a1.info, dict)
    assert isinstance(a2.info, dict)

    a1.evolve((e1,), 0, 0.1)
    a2.evolve((e2,), 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)

    # test copying of the actor
    a3 = a2.copy()
    a3.evolve((e3,), 0, 0.1)
    np.testing.assert_array_equal(e1.data, e3.data)

    a1.make_evolver_numba((e1,))((e1._data_numba,), 0, 0.1)
    a2.make_evolver_numba((e2,))((e2._data_numba,), 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)


@pytest.mark.parametrize("meanfield", [True, False])
def test_local_reactions(meanfield):
    """Test basic methods of the simple reactions actor."""
    if meanfield:
        element = MeanfieldElement(1, {"bounds": [[0, 3]]})
        assert element.concentration == 1
    else:
        element = ScalarFieldElement(1, {"grid": UnitGrid([3])})
    assert np.allclose(element.data, 1)
    assert element.total_amount == 3

    parameters = {"reaction_flux": "2 + 1 * c + t"}
    actor = LocalReactionsActor(parameters=parameters)
    assert isinstance(actor.info, dict)
    assert actor.num_elements == 1
    assert 0 < actor.estimate_dt(element) < 1

    # numpy version
    actor.evolve((element,), 0, dt=1)
    assert np.allclose(element.data, 4)

    actor.evolve((element,), 1, dt=1)
    assert np.allclose(element.data, 11)
    actor.evolve((element,), 1, dt=0)
    assert np.allclose(element.data, 11)

    # numba version
    element.data[...] = 1
    evolver = actor.make_evolver_numba((element,))
    evolver((element._data_numba,), 0, dt=1)
    assert np.allclose(element.data, 4)

    evolver((element._data_numba,), 1, dt=1)
    assert np.allclose(element.data, 11)
    evolver((element._data_numba,), 1, dt=0)
    assert np.allclose(element.data, 11)


@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `phasesep` module"
)
def test_diffusion_vs_reaction_diffusion(rng):
    """Compare the diffusion background with the RD-background."""
    field = ScalarField.random_uniform(UnitGrid([10]), rng=rng)
    e1 = ScalarFieldElement.from_field(field)
    e2 = e1.copy(method="data")

    a1 = DiffusionActor()
    a2 = ReactionDiffusionActor()

    a1.evolve((e1,), 0, 0.1)
    a2.evolve((e2,), 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)

    a1.make_evolver_numba((e1,))((e1._data_numba,), 0, 0.1)
    a2.make_evolver_numba((e2,))((e2._data_numba,), 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)


@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `phasesep` module"
)
def test_reaction_diffusion_background(rng):
    """Test a diffusion background with a reaction."""
    field = ScalarField.random_uniform(UnitGrid([10]), rng=rng)
    element = ScalarFieldElement.from_field(field)
    actor = ReactionDiffusionActor(parameters={"reaction_flux": "-c"})
    dt = actor.estimate_dt((element,))

    for _ in range(100):
        actor.evolve((element,), 0, dt)

    np.testing.assert_allclose(element.data, 0, atol=1e-4)


def test_collection_pde(rng):
    """Test CollectionPDEActor."""
    grid = UnitGrid([10])
    fields = FieldCollection.scalar_random_uniform(2, grid, rng=rng)
    eqs = PDE({"a": "laplace(a)", "b": "2*laplace(b)"})
    truth = eqs.solve(fields, t_range=1, dt=0.1, backend="numpy", tracker=None)

    # test numpy implementation
    element = FieldCollectionElement.from_fields(fields)
    element2 = element.copy(method="data")
    assert element.num_fields == 2
    assert element.degrees_of_freedom == 20
    actor = CollectionPDEActor(eqs)
    for _ in range(10):
        actor.evolve((element,), 0, 0.1)

    np.testing.assert_allclose(element.field.data, truth.data)

    # test numba implementation
    evolve = actor.make_evolver_numba((element2,))
    for _ in range(10):
        evolve((element2._data_numba,), 0, 0.1)

    np.testing.assert_allclose(element2.field.data, truth.data)
