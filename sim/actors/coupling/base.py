'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


from typing import Tuple, Type
from abc import ABCMeta

from pde.tools.misc import classproperty

from ..base import ActorBase
from ...elements.base import ElementBase



class CouplingActorBase(ActorBase, metaclass=ABCMeta):
    """ represents the dynamics of many agents of the same type """
    
    state_classes: Tuple[Type[ElementBase], Type[ElementBase]]
    
    @classproperty
    def num_elements(self) -> int:  # type: ignore
        """ int: the number of elements this actor affects """
        return len(self.state_classes)
