"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np


def recarrays_allclose(a: np.ndarray, b: np.ndarray, **kwargs) -> bool:
    """tests whether the entries of two structured arrays are all close"""
    if a.dtype != b.dtype:
        return False
    return all(np.allclose(a[name], b[name], **kwargs) for name in a.dtype.names)


def assert_recarrays_allclose(a: np.ndarray, b: np.ndarray, **kwargs) -> None:
    """tests whether the entries of two structured arrays are all close"""
    assert a.dtype == b.dtype, "Dtypes of arrays do not match"
    for name in a.dtype.names:
        np.testing.assert_allclose(a[name], b[name], **kwargs)
