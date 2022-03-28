"""
Provides actors that determine the dynamics of the simulation by modifying the
state of elements in time. The actors are separated into several categories, which we
describe separately below.


**General actors** are classes that provide basic infrastructure to implement custom
implementations: 

.. autosummary::
   :nosignatures:

   
   ~base.ActorBase
   ~function.FunctionActor
   ~function.NumbaFunctionActor
   

**Autonomous actors** only affect a single element and thus describe the autonomous
dynamics of this element when it is not coupled to other elements. 

.. autosummary::
   :nosignatures:

   ~autonomous.active_particles.ActiveParticleActor
   ~autonomous.box.BoxActor
   ~autonomous.brownian_motion.BrownianMotionActor
   ~autonomous.coalescence.CoalescenceDropletActor
   ~autonomous.emitters.EmittersActor 
   ~autonomous.fields.MeanfieldActor
   ~autonomous.fields.ScalarPDEActor
   ~autonomous.fields.DiffusionActor
   ~autonomous.fields.ReactionDiffusionActor
   ~autonomous.fields.CollectionPDEActor


**Coupling actors** affect several elements and thus describe a coupling between these
elements.  

.. autosummary::
   :nosignatures:

   ~coupling.fields.FieldCouplingActor
   ~coupling.point_droplet.PointDropletActor
   ~coupling.spherical_droplet.SphericalDropletActor
   

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .autonomous import *
from .base import ActorBase
from .coupling import *
from .function import FunctionActor, NumbaFunctionActor
