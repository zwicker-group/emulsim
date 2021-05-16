"""
Provides classes representing elements of a simulation

.. autosummary::
   :nosignatures:

   ~fields.ScalarFieldElement
   ~fields.MeanfieldElement
   ~points.PointsElement
   ~points.ArrowsElement
   ~spherical_droplets.SphericalDropletsElement

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import ElementBase
from .fields import FieldElementBase, MeanfieldElement, ScalarFieldElement
from .points import ArrowsElement, PointsElement

try:
    from .spherical_droplets import SphericalDropletsElement
except ImportError:
    pass
