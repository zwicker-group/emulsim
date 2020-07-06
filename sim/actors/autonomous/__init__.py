"""
Provides actors that affect single elements in a simulation

.. autosummary::
   :nosignatures:

   ~fields.MeanfieldActor
   ~fields.ScalarPDEActor
   ~fields.DiffusionActor
   ~fields.ReactionDiffusionActor
   ~emitters.EmittersActor 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .emitters import EmittersActor
from .fields import (
    DiffusionActor,
    MeanfieldActor,
    ReactionDiffusionActor,
    ScalarPDEActor,
)
