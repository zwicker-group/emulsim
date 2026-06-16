"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from droplets import Emulsion
from pde.backends.numba.utils import jit

from emulsim.elements import MulticomponentDroplet, MulticomponentDropletsElement


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_multicomponent_droplets(dim, num_comps):
    """Test basic multicomponent droplets functions."""
    emulsion = [
        MulticomponentDroplet([0] * dim, 1, amounts=[1] * num_comps),
        MulticomponentDroplet([1] * dim, 2, amounts=[2] * num_comps),
    ]
    element = MulticomponentDropletsElement.from_droplets(emulsion, copy=True)
    assert element.dim == dim
    assert element.num_comps == num_comps
    assert element.degrees_of_freedom == 2 * (dim + num_comps + 1)
    np.testing.assert_allclose(element.amounts, np.full(num_comps, 3))

    # generic plot test
    if dim == 2:
        element.plot()

    # test merging
    emulsion[0].merge(emulsion[1], inplace=True)
    np.testing.assert_allclose(emulsion[0].amounts, np.full(num_comps, 3))

    # test merging using numba
    d1 = MulticomponentDroplet([0] * dim, 1, amounts=[1] * num_comps)
    d2 = MulticomponentDroplet([2] * dim, 1, amounts=[2] * num_comps)
    d3 = d1.copy()

    merge_data = jit(MulticomponentDroplet._make_merge_data())
    merge_data(d1.data, d2.data, out=d3.data)
    np.testing.assert_allclose(d3.position, [1] * dim)
    np.testing.assert_allclose(d3.amounts, [3] * num_comps)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_comps", [1, 2])
def test_empty_multicomponent_droplets(dim, num_comps):
    """Test empty MulticomponentDropletsElement."""
    d = MulticomponentDroplet([0] * dim, 1, amounts=[1] * num_comps)

    def check(el, size, num_comps=num_comps):
        assert el.droplet_count == 0
        assert el.data.size == size
        assert el.dim == dim
        assert el.num_comps == num_comps

    # really empty
    el = MulticomponentDropletsElement.from_droplets(Emulsion([], dtype=d.data.dtype))
    check(el, 0)

    el = MulticomponentDropletsElement.empty(0, droplet=d)
    check(el, 0)

    el = MulticomponentDropletsElement.empty(0, dim=dim)
    check(el, 0, num_comps=1)

    # additional space
    el = MulticomponentDropletsElement.from_droplets(
        Emulsion([], dtype=d.data.dtype), maxcount=3
    )
    check(el, 3)

    el = MulticomponentDropletsElement.empty(3, droplet=d)
    check(el, 3)

    el = MulticomponentDropletsElement.empty(3, dim=dim)
    check(el, 3, num_comps=1)
