'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from pde import UnitGrid, CartesianGrid, ScalarField, DiffusionPDE
from pde.tools.misc import skipUnlessModule

from ....elements import ScalarFieldElement, MeanfieldElement
from ..fields import MeanfieldActor, ScalarPDEActor, DiffusionActor, ReactionDiffusionActor



def test_diffusion_actor():
    """ test basic methods of the simple diffusion actor """
    grid = CartesianGrid([[0, 10]], 5, periodic=True)
    element = ScalarFieldElement.from_field(ScalarField(grid, 3))
    assert element.grid == grid
    actor = DiffusionActor()
    assert isinstance(actor.estimate_dt(element), float)
    assert actor.num_elements == 1
    
    

def test_diffusion_vs_pde():
    """ compare the diffusion background with the PDF actor """
    field = ScalarField.random_uniform(UnitGrid([10]))
    e1 = ScalarFieldElement.from_field(field)
    e2 = e1.copy()

    a1 = DiffusionActor()
    a2 = ScalarPDEActor(DiffusionPDE())
    assert isinstance(a1.info, dict)
    assert isinstance(a2.info, dict)
    
    a1.evolve(e1, 0, 0.1)
    a2.evolve(e2, 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)
    
    a1.make_evolver_numba(e1)(e1.data, 0, 0.1)
    a2.make_evolver_numba(e2)(e2.data, 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)
    
    

@skipUnlessModule("phasesep")
def test_meanfield_reactions():
    """ test basic methods of the simple mean field background """
    element = MeanfieldElement(1, {'bounds': [[0, 3]]})
    assert element.concentration == 1
    assert element.total_amount == 3
    
    parameters = {'reaction_flux': '2 + 1 * c + t'}
    actor = MeanfieldActor(parameters=parameters)
    assert isinstance(actor.info, dict)
    assert actor.num_elements == 1
    assert 0 < actor.estimate_dt(element) < 1
    
    actor.evolve(element, 0, dt=1)
    assert element.concentration == pytest.approx(4)
    
    actor.evolve(element, 1, dt=1)
    assert element.concentration == pytest.approx(11)
    actor.evolve(element, 1, dt=0)
    assert element.concentration == pytest.approx(11)
    


@skipUnlessModule("phasesep")
def test_diffusion_vs_reaction_diffusion():
    """ compare the diffusion background with the RD-background """
    field = ScalarField.random_uniform(UnitGrid([10]))
    e1 = ScalarFieldElement.from_field(field)
    e2 = e1.copy()

    a1 = DiffusionActor()
    a2 = ReactionDiffusionActor()
    
    a1.evolve(e1, 0, 0.1)
    a2.evolve(e2, 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)
    
    a1.make_evolver_numba(e1)(e1.data, 0, 0.1)
    a2.make_evolver_numba(e2)(e2.data, 0, 0.1)
    np.testing.assert_array_equal(e1.data, e2.data)
    
        
    
@skipUnlessModule("phasesep")  
def test_reaction_diffusion_background():
    """ test a diffusion background with a reaction """
    field = ScalarField.random_uniform(UnitGrid([10]))
    element = ScalarFieldElement.from_field(field)
    actor = ReactionDiffusionActor(parameters={'reaction_flux': '-c'})
    dt = actor.estimate_dt(element)
    
    for _ in range(100):
        actor.evolve(element, 0, dt)
        
    np.testing.assert_allclose(element.data, 0, atol=1e-4)
    
