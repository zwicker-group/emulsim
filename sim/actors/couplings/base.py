'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


from typing import Tuple  # @UnusedImport
from abc import ABCMeta

from ..base import ActorBase
from ...elements.base import ElementBase



class CouplingActorBase(ActorBase, metaclass=ABCMeta):
    """ represents the dynamics of many agents of the same type """
    
    num_elements: int = 2
    state_classes: Tuple[ElementBase, ElementBase]
