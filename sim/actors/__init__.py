"""Provides actors that determine the dynamics of the simulation by modifying the state
of elements during each time step. Actors are separated into several categories, which
we describe separately below.

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
   ~autonomous.fields.LocalReactionsActor
   ~autonomous.fields.ScalarPDEActor
   ~autonomous.fields.DiffusionActor
   ~autonomous.fields.ReactionDiffusionActor
   ~autonomous.fields.CollectionPDEActor


**Coupling actors** affect several elements and thus describe a coupling between these
elements.

.. autosummary::
   :nosignatures:

   ~coupling.nucleation.DropletNucleationActor
   ~coupling.fields.FieldCouplingActor
   ~coupling.fields.FieldExchangeActor
   ~coupling.fields.FieldBoundaryExchangeActor
   ~coupling.point_droplet.PointDropletActor
   ~coupling.spherical_droplet.SphericalDropletActor
   ~coupling.multicomponent_droplet.MulticomponentDropletActor


Use :func:`~base.find_actors` to discover actors that are compatible with a given list
of elements.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .autonomous import *
from .base import ActorBase, find_actors
from .coupling import *
from .function import FunctionActor, NumbaFunctionActor
