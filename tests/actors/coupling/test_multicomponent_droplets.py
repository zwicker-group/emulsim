"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured as s2u

import pde
from pde.tools.misc import module_available
from pde.tools.numba import jit

from sim import Simulation, State
from sim.actors.coupling import MulticomponentDropletActor
from sim.actors.coupling.multicomponent_droplet import _make_regularizer
from sim.elements import (
    FieldCollectionElement,
    MulticomponentDroplet,
    MulticomponentDropletsElement,
)


@pytest.mark.parametrize("do_jit", [True, False])
def test_make_regularizer(do_jit):
    """Test regularizer."""
    jitter = jit if do_jit else lambda x: x

    a = np.array([0.48247728, 0.52013119])
    _make_regularizer(2)(a)
    assert sum(a) < 1

    a = np.array([0.48247728, 0.52013119])
    jitter(_make_regularizer(2))(a)
    assert sum(a) < 1

    test_arr = np.array([[-0.1, 0.0, 0.6], [0.5, 1.1, 0.6]])
    res_arr = np.array([[0, 0.0, 0.5], [0.5, 1, 0.5]])

    regularize = jitter(_make_regularizer(2, eps=0))
    phis = test_arr.copy()
    regularize(phis)
    np.testing.assert_allclose(phis, res_arr)

    regularize = jitter(_make_regularizer(2, eps=0))
    for i in range(len(test_arr)):
        phis = test_arr[:, i].copy()
        regularize(phis)
        np.testing.assert_allclose(phis, res_arr[:, i])

    regularize = jitter(_make_regularizer(2, eps=0.1))
    phis = np.array([[-0.1, 0.0, 0.6], [0.5, 1.1, 0.6]])
    regularize(phis)
    np.testing.assert_allclose(phis, np.array([[0.1, 0.1, 0.45], [0.5, 0.8, 0.45]]))


@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `phasesep` module"
)
def test_multicomponent_thermodynamics():
    """Test the implementation of the thermodynamics."""
    from phasesep import FloryHuggins2Components

    chi = 3
    droplet_actor = MulticomponentDropletActor({"chis": [[0]], "chis_solvent": chi})
    fFH = FloryHuggins2Components(chi=chi)

    cs = np.linspace(0, 1, 128)[1:-1]
    calc_state = droplet_actor._make_calc_state_vars()
    fs = np.ravel([calc_state(c)[0] for c in cs.reshape(-1, 1)])
    mus = np.ravel([calc_state(c)[1] for c in cs.reshape(-1, 1)])
    ps = np.ravel([calc_state(c)[2] for c in cs.reshape(-1, 1)])

    np.testing.assert_allclose(fs, fFH(cs))
    np.testing.assert_allclose(mus, fFH.chemical_potential(cs))
    np.testing.assert_allclose(ps, fFH.pressure(cs))


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_multicomponent_droplet_actor(dim, num_comps, rng):
    """Test basic multicomponent droplets simulations."""
    # create the background field
    grid = pde.CartesianGrid([[0, 32]] * dim, 1, periodic=True)
    fc = pde.FieldCollection.scalar_random_uniform(
        num_comps, grid, 0, 0.1 / num_comps, rng=rng
    )
    bulk = FieldCollectionElement.from_fields(fc)

    # create some droplets
    drop_list = [
        MulticomponentDroplet.from_composition(
            grid.get_random_point(rng=rng),  # position
            rng.uniform(1, 2),  # radius
            rng.uniform(0, 0.9 / num_comps, num_comps),  # composition
        )
        for _ in range(3)
    ]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)

    # create the simulation state
    state = State({"bulk": bulk, "droplets": droplets_element})
    amounts = state.get_total_quantity("amounts")

    # create the dynamics
    simulation = Simulation(state)
    droplet_actor = MulticomponentDropletActor(
        {"chis": np.full(num_comps, 1), "chis_solvent": 3}
    )
    simulation.add_actor(("droplets", "bulk"), droplet_actor)

    res1 = simulation.run(t_range=10, backend="numpy", dt=1e-2, tracker=None)
    res2 = simulation.run(t_range=10, backend="numba", dt=1e-2, tracker=None)

    np.testing.assert_allclose(
        s2u(res1["droplets"].data), s2u(res2["droplets"].data), atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(res1["bulk"].data, res2["bulk"].data, atol=1e-4)

    # check for material conservation
    np.testing.assert_allclose(res1.get_total_quantity("amounts"), amounts)
    np.testing.assert_allclose(res2.get_total_quantity("amounts"), amounts)


# @pytest.mark.parametrize("dim", [1, 3])
# @pytest.mark.parametrize("num_comps", [1, 2])
# def test_multicomponent_no_droplets(dim, num_comps):
#     """test basic multicomponent droplets simulations"""
#     # create the background field
#     grid = pde.CartesianGrid([[0, 32]] * dim, 1, periodic=True)
#     fc = pde.FieldCollection.scalar_random_uniform(num_comps, grid, 0, 0.1 / num_comps)
#     bulk = FieldCollectionElement.from_fields(fc)
#
#     # create no droplets :)
#     dtype = MulticomponentDroplet.get_dtype(
#         amounts=np.zeros(num_comps), position=np.zeros(dim)
#     )
#     data = np.empty((0,), dtype=dtype)
#     droplets_element = MulticomponentDropletsElement(data)
#
#     # create the simulation state
#     state = State({"bulk": bulk, "droplets": droplets_element})
#     amounts = state.get_total_quantity("amounts")
#
#     # create the dynamics
#     simulation = Simulation(state)
#     droplet_actor = MulticomponentDropletActor(
#         {"chis": np.full(num_comps, 1), "chis_solvent": 3}
#     )
#     simulation.add_actor(("droplets", "bulk"), droplet_actor)
#
#     res1 = simulation.run(t_range=1, backend="numpy", dt=1e-2, tracker=None)
#     res2 = simulation.run(t_range=1, backend="numba", dt=1e-2, tracker=None)
#
#     for res in [res1, res2]:
#         assert len(res["droplets"]) == 0
#         np.testing.assert_allclose(res["bulk"].data, state["bulk"].data)
#         np.testing.assert_allclose(res.get_total_quantity("amounts"), amounts)


def test_multicomponent_equilibrium():
    """Test equilibrium in multicomponent system."""
    grid = pde.CartesianGrid([[0, 32]] * 3, 1, periodic=True)
    fc = pde.FieldCollection.from_scalar_expressions(grid, [0.05, 0.15])

    bulk = FieldCollectionElement.from_fields(fc)
    drop_list = [MulticomponentDroplet.from_composition([16] * 3, 2, [0.3, 0.5])]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)
    state = State({"bulk": bulk, "droplets": droplets_element})
    amounts = state.get_total_quantity("amounts")

    simulation = Simulation(state)
    droplet_actor = MulticomponentDropletActor(
        {"chis": [[0, 1], [1, 0]], "chis_solvent": 3, "mobility": [1, 2]}
    )
    simulation.add_actor(("droplets", "bulk"), droplet_actor)

    result = simulation.run(t_range=1e4, dt=0.1, tracker=None)

    # compare between elements
    np.testing.assert_allclose(result.get_total_quantity("amounts"), amounts)

    mu_d, mu_f = droplet_actor.get_thermodynamic_quantity(
        result["droplets"], result["bulk"], kind="chemical potential"
    )
    assert mu_d[0][0] == pytest.approx(mu_f[0].average)
    assert mu_d[0][1] == pytest.approx(mu_f[1].average)

    p_d, p_f = droplet_actor.get_thermodynamic_quantity(
        result["droplets"], result["bulk"], kind="pressure"
    )
    assert p_d[0] == pytest.approx(p_f.average)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_multicomponent_coexistence(backend):
    """Test coexistence in multicomponent system."""
    grid = pde.CartesianGrid([[0, 32]] * 3, 1, periodic=True)
    fc = pde.FieldCollection.from_scalar_expressions(grid, [0.1])

    bulk = FieldCollectionElement.from_fields(fc)
    drop_list = [MulticomponentDroplet.from_composition([16] * 3, 1, [0.8])]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)
    state = State({"bulk": bulk, "droplets": droplets_element})
    amount = state.get_total_quantity("amounts")[0]

    simulation = Simulation(state)
    chi = 3.0
    droplet_actor = MulticomponentDropletActor({"chis": [[0]], "chis_solvent": chi})
    simulation.add_actor(("droplets", "bulk"), droplet_actor)

    result = simulation.run(t_range=1000, backend=backend, dt=0.1, tracker=None)

    # compare to theory
    phis = droplet_actor.get_droplet_fractions((result["droplets"], result["bulk"]))
    phiOut, phiIn = phis[0, :, 0]
    assert result.get_total_quantity("amounts")[0] == pytest.approx(amount)
    chiIn_equivalent = np.log(phiIn / (1 - phiIn)) / (2 * phiIn - 1)
    assert chiIn_equivalent == pytest.approx(chi, rel=0.1)
    chiOut_equivalent = np.log(phiOut / (1 - phiOut)) / (2 * phiOut - 1)
    assert chiOut_equivalent == pytest.approx(chi, rel=0.1)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_multicomponent_coarsening(backend, rng):
    """Simple test of coarsening of multicomponent droplets."""
    grid = pde.CartesianGrid([[-64, 64]] * 3, 2, periodic=True)

    phiOut = 0.1
    phiIn = 1 - phiOut
    chi = np.log(phiOut / (1 - phiOut)) / (2 * phiOut - 1)

    drop_list = [
        MulticomponentDroplet.from_composition(
            grid.get_random_point(rng=rng), 0.5, [phiIn]
        ),
        MulticomponentDroplet.from_composition(
            grid.get_random_point(rng=rng), 1, [phiIn]
        ),
    ]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)
    fc = pde.FieldCollection.from_scalar_expressions(grid, [phiOut])
    bulk = FieldCollectionElement.from_fields(fc)
    state = State({"droplets": droplets_element, "bulk": bulk})
    amount = state.get_total_quantity("total_amount")

    simulation = Simulation(state)
    droplet_actor = MulticomponentDropletActor(
        parameters={"chis": [[0]], "chis_solvent": chi, "surface_tension": 0.1}
    )
    simulation.add_actor(("droplets", "bulk"), droplet_actor)

    result = simulation.run(t_range=1, backend=backend, dt=0.1, tracker=None)
    assert result.get_total_quantity("total_amount") == pytest.approx(amount)
    assert result["droplets"].data[0]["radius"] < 0.5
    assert result["droplets"].data[1]["radius"] > 1


@pytest.mark.skipif(
    not module_available("droplets"), reason="requires `droplets` module"
)
@pytest.mark.skipif(
    not module_available("phasesep"), reason="requires `phasesep` module"
)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_multicomponent_active_droplet(backend):
    """Test active droplet simulation."""
    import droplets
    import phasesep

    chi, mobility, kf, kb, t_range = 3.0, 1, 0.002, 0.01, 1000

    # run py-sim simulation
    grid = pde.CartesianGrid([[-64, 64]] * 3, 1, periodic=True)
    fc = pde.FieldCollection.from_scalar_expressions(grid, [0.1])

    bulk = FieldCollectionElement.from_fields(fc)
    drop_list = [MulticomponentDroplet.from_composition([16] * 3, 1, [0.8])]
    droplets_element = MulticomponentDropletsElement.from_droplets(drop_list)
    state = State({"bulk": bulk, "droplets": droplets_element})

    simulation = Simulation(state)
    droplet_actor = MulticomponentDropletActor.from_linear_reactions(
        parameters={"chis": [[0]], "chis_solvent": chi, "mobility": mobility},
        rates=[-kb],
        production=kf,
    )
    simulation.add_actor(("droplets", "bulk"), droplet_actor)

    result = simulation.run(t_range=t_range, backend=backend, dt=0.1, tracker=None)

    # run py-phasesep simulation
    grid_sph = pde.SphericalSymGrid(80, 100)
    drop_sph = droplets.DiffuseDroplet([0, 0, 0], 4, 1)
    field_sph = drop_sph.get_phase_field(grid_sph, vmin=0.1, vmax=0.9)

    eq = phasesep.CahnHilliardExtendedPDE(
        {
            "free_energy": phasesep.FloryHuggins2Components(chi=chi),
            "reaction_flux": f"{kf} - {kb} * c",
            "mobility": f"{mobility} * c * (1 - c)",
        }
    )
    res_sph = eq.solve(
        field_sph, t_range=t_range, dt=0.0001, adaptive=True, tracker=None
    )
    drop_sph = droplets.locate_droplets(res_sph, refine=True)

    # comparison
    drop1 = result["droplets"].droplets[0]
    phis = droplet_actor.get_droplet_fractions((result["droplets"], result["bulk"]))
    phi1_out, phi1_in = phis[0, :, 0]

    assert drop1.radius == pytest.approx(drop_sph[0].radius, rel=0.2)
    assert phi1_out == pytest.approx(res_sph.data[-1], rel=0.01)
    assert phi1_in == pytest.approx(res_sph.data[0], rel=0.1)
