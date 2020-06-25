'''
Test generic elements functionality

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from pde import UnitGrid, ScalarField
from pde.tools.misc import skipUnlessModule
from droplets import SphericalDroplet

from .. import PointsElement, SphericalDropletsElement, MeanfieldElement, ScalarFieldElement, element_from_file



def generate_elements():
    """ helper function generating all tested backgrounds """
    yield PointsElement(np.random.randn(3, 1))  # 1d
    yield PointsElement(np.random.randn(3, 2))  # 2d
    
    emulsion = [SphericalDroplet([0], 1), SphericalDroplet([1], 2)]
    yield SphericalDropletsElement.from_droplets(emulsion)
    emulsion = [SphericalDroplet([0, 0], 1), SphericalDroplet([1, 1], 2)]
    yield SphericalDropletsElement.from_droplets(emulsion)
    
    yield MeanfieldElement(0.1, {'bounds': [[0, 1]]})  # 1d 
    yield MeanfieldElement(0.1, {'bounds': [[0, 1], [0, 1]]})  # 2d 
    
    field = ScalarField.random_normal(UnitGrid([3, 3]))
    yield ScalarFieldElement.from_field(field)


    
@pytest.mark.parametrize("element", generate_elements())
def test_basic(element):
    """ test basic functions of elements """
    assert isinstance(str(element), str)
    assert isinstance(repr(element), str)
    assert isinstance(element.attributes, dict)
    assert isinstance(element.data, np.ndarray)
    
    e1 = element.copy()
    assert e1 is not element
    assert e1 == element
    
    # test generic plotting
    if isinstance(element, PointsElement) or element.dim == 2:
        element.plot(action='close')



@skipUnlessModule('h5py')
@pytest.mark.parametrize("element", generate_elements())
def test_element_io(element, tmp_path):
    """ test writing and reading agents states """
    if isinstance(element, ScalarFieldElement):
        return
        
    path = tmp_path / f"test_io_{element.__class__.__name__}.hdf"
    
    element.to_file(path)
    element2 = element_from_file(path)
    assert element == element2
    assert element is not element2
        