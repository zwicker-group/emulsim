'''
Module defining the abstract base class of actors

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Any # @UnusedImport
from abc import ABCMeta

from ..base import ActorBase
from ...elements.base import ElementBase



class AutonomousActorBase(ActorBase, metaclass=ABCMeta):
    """ represents an actor affecting a single element """
    
    num_elements: int = 1
    state_class: Any = ElementBase
