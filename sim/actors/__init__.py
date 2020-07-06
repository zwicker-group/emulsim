"""
Provides actors that determine the dynamics of the simulation by modifying the
state of elements in time.

.. autosummary::
   :nosignatures:

   ActorBase
   ~autonomous.fields.MeanfieldActor
   ~autonomous.fields.ScalarPDEActor
   ~autonomous.fields.DiffusionActor
   ~autonomous.fields.ReactionDiffusionActor
   ~autonomous.emitters.EmittersActor 
   ~coupling.point_droplet.PointDropletActor
   ~coupling.spherical_droplet.SphericalDropletActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .autonomous import *
from .base import ActorBase
from .coupling import *
