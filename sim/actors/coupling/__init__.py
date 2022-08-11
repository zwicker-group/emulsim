"""
Provides actors that affect multiple elements in a simulation.

.. autosummary::
   :nosignatures:

   ~fields.FieldCouplingActor
   ~fields.FieldBoundaryExchangeActor
   ~point_droplet.PointDropletActor
   ~spherical_droplet.SphericalDropletActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .fields import FieldBoundaryExchangeActor, FieldCouplingActor
from .point_droplet import PointDropletActor
from .spherical_droplet import SphericalDropletActor
