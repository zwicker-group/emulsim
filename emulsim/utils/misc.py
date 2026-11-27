"""Miscellaneous functions

.. autosummary::
   :nosignatures:

   ~init_zarr_storage_format

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

_SET_ZARR_STORAGE_FORMAT: bool = False


def init_zarr_storage_format():
    """Sets default zarr storage format to version 2 to support structured arrays.

    The default format version is only changed at the first call of this function!
    """
    global _SET_ZARR_STORAGE_FORMAT
    if _SET_ZARR_STORAGE_FORMAT:
        return
    try:
        from modelrunner.storage.backend.zarr import ZarrStorage
    except ImportError:
        pass  # zarr storage is not available
    else:
        ZarrStorage.default_zarr_format = 2
    _SET_ZARR_STORAGE_FORMAT = True
