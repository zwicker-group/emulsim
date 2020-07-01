'''
Provides classes for acting on the elements in a simulation. These classes
define the dynamics of a simulation.

.. autosummary::
   :nosignatures:

   ~autonomous.fields.MeanfieldActor
   ~autonomous.fields.ScalarPDEActor
   ~autonomous.fields.DiffusionActor
   ~autonomous.fields.ReactionDiffusionActor
   ~autonomous.emitters.EmittersActor 
   ~coupling.point_droplet.PointDropletActor
   ~coupling.spherical_droplet.SphericalDropletActor

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from .autonomous import *
from .coupling import *