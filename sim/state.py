'''
Provides a class representing the full system state of multiple elements

.. autosummary::
   :nosignatures:

   ~State

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import logging
from typing import Any, Dict, Union, Sequence
from collections import defaultdict, OrderedDict

from pde.grids.base import DimensionError
from pde.tools.misc import hdf_write_attributes
from pde.tools.plotting import plot_on_axes

from .elements.base import ElementBase



class State():
    """ Class defining the state of the agent-based simulation """
    
#     @classmethod
#     def from_hdf_dataset(cls, dataset) -> "SimulationState":
#         """ construct the instance by reading data from an hdf5 dataset
#         
#         Args:
#             dataset: the hdf5 dataset (in an already opened file)
#         """
#         background = background_state_from_hdf(dataset['background'])
#         agents = agents_state_from_hdf(dataset['agents'])
#         return cls(background, agents) 
#     
#     
#     @classmethod
#     def from_file(cls, path: str) -> "SimulationState":
#         """ create simulation state instance from data stored in a hdf file
#          
#         Args:
#             path (str): Path to the hdf file being read
#         """
#         import h5py
#         
#         with h5py.File(path, "r") as fp:
#             return cls.from_hdf_dataset(fp)

    def __init__(self, elements: Dict[str, ElementBase] = None):
        self._logger = logging.getLogger(__name__)
        self.elements = OrderedDict()
        self.dim = None
        if elements:
            for name, element in elements.items():
                self.add_element(name, element)


    def add_element(self, name: str, element: ElementBase):
        if name in self.elements:
            self._logger.warning('Overwriting element `%s` in state', name)
        if len(self.elements) == 0:
            self.dim = element.dim
        elif self.dim != element.dim:
            raise DimensionError(f'Dimension of element ({element.dim}) differs '
                                 f'from state ({self.dim})')
        self.elements[name] = element


    def get_index(self, name: str) -> int:
        for i, element_name in enumerate(self.elements):
            if name == element_name:
                return i
        raise KeyError(f'`{name}` not in {self.__class__.__name__}') 


    def __getitem__(self, key: Union[str, Sequence[str]]):
        if isinstance(key, str):
            return self.elements[key]
        else:
            return [self.elements[k] for k in key]
        
        
    def __len__(self) -> int:
        return len(self.elements)
    
    
    def __iter__(self):
        return iter(self.elements.items())
        
        
    def __contains__(self, name: str):
        return name in self.elements

        
    def __str__(self):
        elements_str = ', '.join(f'"{name}": {element!s}'
                                 for name, element in self)
        return f"{self.__class__.__name__}({{{elements_str}}})"
        
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.elements!r})"
        

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(self.elements[key] == other.elements[key]
                   for key in self.elements)
        
        
    def copy(self) -> "State":
        """ copy the state """
        return self.__class__({name: element.copy()
                               for name, element in self})
        

    @property
    def attributes(self) -> Dict[str, Any]:
        """ dict: information about the state """
        return {'elements': {name: element.attributes
                             for name, element in self}}
        
        
    def _write_hdf_dataset(self, hdf_path):
        """ write data to a given hdf5 file pointer `hdf_path` """
        for name, element in self.elements.items():
            element._write_hdf_dataset(hdf_path.create_group(name))


    def to_file(self, filename: str, info: Dict[str, Any] = None) -> None:
        r""" store agents state in a hdf file
        
        Args:
            filename (str):
                Path where the data is stored
            info (dict):
                Extra information that is written to the hdf attributes. Note
                that the values in this dictionary will be JSON-serialized. 
        """
        import h5py
        with h5py.File(filename, "w") as fp:
            self._write_hdf_dataset(fp)
            hdf_write_attributes(fp, info)            
            
        
    @property
    def data(self) -> Dict[str, Any]:
        """ the full data of the simulation """
        return tuple(element.data for element in self.elements.values())
        
             
    @plot_on_axes()
    def plot(self, ax,
             element_args: Dict[str, Any] = None,
             **kwargs):
        r""" plot the emulsion together with the background field
         
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes in which the simulation state is shown.
            elements (str):
                Determines which elements are plotted. Possible values are
                `all`, `droplets`, or `background`.
            background_args (dict):
                Additional arguments for the background plot
            agent_args (dict):
                Additional arguments for the agents plot
        """
        if element_args:
            element_args = defaultdict(dict, element_args)
        else:
            element_args = defaultdict(dict)
             
        for name, element in self:
            element.plot(ax=ax, **element_args[name], **kwargs)
