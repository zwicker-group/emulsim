'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Dict, Any

import numpy as np

from pde.tools.parameters import Parameter
from pde.tools.plotting import plot_on_axes

from .base import ElementBase



class PointsElement(ElementBase):
    """ represents the state of agents that emit mass into the background """

    parameters_default = [
        Parameter('representative_radius', 1, float,
                  "Radius used for representing the point")]


    def __init__(self, data: np.ndarray = None,
                 parameters: Dict[str, Any] = None):
        """
        Args:
            positions (:class:`numpy.ndarray`):
                The positions of all points
            parameters (dict):
                Additional parameters. Call
                :meth:`~PointsElement.show_parameters` for details.
        """
        # initialize parameters
        super().__init__(data, parameters)
        
        # ensure the right format of the input data
        self.data = np.atleast_2d(data)
        if self.data.ndim != 2:
            raise ValueError('`positions` must be a sequence of positions')
        self.dim = self.data.shape[1]

    
    def __len__(self) -> int:
        return len(self.data)


    @plot_on_axes()
    def plot(self, ax, color='red', **kwargs):
        """ plot all emitter agents
        
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes into which the agents are plotted
            color (matplotlib color):
                The color with which emitters are shown
        """
        import matplotlib as mpl
        
        if self.dim == 1:
            positions = np.c_[np.zeros(len(self)), self.data]
        elif self.dim == 2:
            positions = self.data
        else:
            raise RuntimeError(f"Cannot plot points with dimension {self.dim}")
        
        # create the patches
        radius = self.parameters['representative_radius']
        patches = [mpl.patches.Circle(pos, radius) for pos in positions]
        
        # add all patches as a collection
        coll = mpl.collections.PatchCollection(patches, facecolors=(color,))
        ax.add_collection(coll)

