"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from .. import MulticomponentDroplet, MulticomponentDropletsElement


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_multicomponent_droplets(dim, num_comps):
    """test basic multicomponent droplets functions"""
    emulsion = [
        MulticomponentDroplet([0] * dim, 1, amounts=[1] * num_comps),
        MulticomponentDroplet([1] * dim, 2, amounts=[2] * num_comps),
    ]
    element = MulticomponentDropletsElement.from_droplets(emulsion)
    assert element.dim == dim
    assert element.num_comps == num_comps
    assert element.degrees_of_freedom == 2 * (dim + num_comps + 1)
    np.testing.assert_allclose(element.amounts, np.full(num_comps, 3))
