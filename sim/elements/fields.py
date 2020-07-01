'''
Provides elements that represent extended discretized fields 

.. autosummary::
   :nosignatures:

   ~ScalarFieldElement
   ~MeanfieldElement

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from abc import abstractmethod, abstractproperty, ABCMeta
from typing import Dict, Any, Sequence, Tuple, Callable

import numpy as np
import numba as nb

from pde.grids.cartesian import CartesianGridBase, GridBase
from pde.fields import ScalarField
from pde.grids import CartesianGrid
from pde.tools.parameters import Parameter
from pde.tools.cuboid import Cuboid
from pde.tools.plotting import plot_on_axes

from .base import ElementBase



class FieldElementBase(ElementBase, metaclass=ABCMeta):
    """ base class for the background of the agent-based simulation """


    def set_bounds(self, bounds: Sequence[Tuple[float, float]]) -> None:
        """ set the boundaries of the background
        
        Args:
            bounds (sequence):
                A sequence of tuples specifying the lower and upper bound for
                each axis. The number of entries sets the space dimension.
        """
        self._cuboid = Cuboid.from_bounds(np.array(bounds, np.double),
                                          mutable=False)
        self.dim: int = self._cuboid.dim
        self.bounds = self._cuboid.bounds
        self.volume = float(self._cuboid.volume)


    @property
    def grid(self) -> CartesianGridBase:
        """ :class:`pde.grids.cartesian.CartesianGrid`: background grid """
        return CartesianGrid(self.bounds, 1)

    
    def plot(self, ax=None, **kwargs):
        """ plot the background field """
        pass


    @abstractproperty
    def total_amount(self) -> float:
        """ float: the total material amount in the background """
        pass
    

    @abstractmethod
    def get_concentration(self, points):
        """ determine concentration at the given points
        
        Args:
            points (:class:`numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        pass
    

    @abstractmethod
    def add_amount(self, point: np.ndarray, amount: float):
        """ add the given amount to the background
        
        Args:
            point (:class:`numpy.ndarray`):
                Point where the amount is added to the background
            amount (float):
                The total amount added to the background
        """
        pass

    
    def make_get_concentration_compiled(self) -> Callable:
        """ get a compiled function for obtaining concentrations
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`), which determines the concentration
            at point `point` given the background state `data`.
        """
        raise NotImplementedError

    
    def make_add_amount_compiled(self) -> Callable:
        """ get a compiled function for adding amount to the background
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`, amount: float), which adds `amount`
            to the background state given by `data` at point `point`.
        """
        raise NotImplementedError


# 
# def background_state_from_hdf(hdf_path) -> FieldElementBase:
#     """ create background state instance from a stored state
#      
#     Args:
#         hdf_path: HDF Path in an already opened file
#     """
#     # a single field is stored in the data
#     dataset = hdf_path[list(hdf_path.keys())[0]]  # retrieve only dataset
#     
#     # determine class
#     class_name = json.loads(dataset.attrs['class'])
#     field_cls = FieldElementBase._subclasses[class_name]
#     
#     # load the instance from hdf
#     return field_cls.from_hdf_dataset(dataset)  # type: ignore
# 
# 
# 
# def background_state_from_file(path: str) -> FieldElementBase:
#     """ create background state instance from a stored state
#      
#     Args:
#         path (str): Path to the file being read
#     """
#     import h5py
#     
#     with h5py.File(path, "r") as fp:
#         return background_state_from_hdf(fp)
#     
    
    

class MeanfieldElement(FieldElementBase):
    """ the state associated with a meanfield background """

    parameters_default = [
        Parameter('bounds', None,
                  description='Sets the box size')
    ]


    def __init__(self, data: float = 0,
                 parameters: Dict[str, Any] = None):
        """ initialize the meanfield background
        
        Args:
            bounds (sequence):
                A sequence of tuples specifying the lower and upper bound for
                each axis. The number of entries sets the space dimension.
            data (float):
                The initial concentration in the background
        """
        super().__init__(data, parameters)
        
        # store data in a mutable 1d-array
        if self.parameters['bounds'] is None:
            raise ValueError('`bounds` need to be specified in parameters')
        else:
            self.set_bounds(self.parameters['bounds'])
        self.data = np.full((1,), data, dtype=np.double)
        

    @property
    def concentration(self) -> float:
        """ float: the concentration in the background """
        return float(self.data[0])
    
    @concentration.setter
    def concentration(self, value: float):
        """ set the background concentration
        
        Args:
            value (float):
                The new concentration
        """
        self.data[0] = value
        

    @property
    def total_amount(self) -> float:
        """ float: the total material amount in the background """
        return self.concentration * self.volume

    @total_amount.setter
    def total_amount(self, amount: float):
        """ set the total material amount in the background
        
        Args:
            amount (float):
                The new total amount
        """
        self.concentration = amount / self.volume


    def __repr__(self):
        return (f'{self.__class__.__name__}(bounds={self.bounds!r}, '
                f'data={self.concentration})')


    def __str__(self):
        return (f'{self.__class__.__name__}(bounds={self.bounds!s}, '
                f'data={self.concentration})')


    @plot_on_axes()
    def plot(self, ax, color='tab:blue', **kwargs):
        """ plot the background field
        
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes in which the background is shown
            color:
                The matplotlib color in which the background is shown
                
        All additional arguments are ignored.
        """
        if self.dim != 2:
            raise RuntimeError('Can only plot data in two dimensions.')
        
        # create the rectangle representing the background
        from matplotlib import patches
        rect = patches.Rectangle(self._cuboid.pos, *self._cuboid.size,
                                 edgecolor='none', facecolor=color)
        ax.add_patch(rect)
        ax.set_xlim(*self.bounds[0])
        ax.set_ylim(*self.bounds[1])


    def get_concentration(self, points: np.ndarray):
        """ determine concentration at the given points
        
        Args:
            points (:class:`numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        points = np.asanyarray(points)
        if points.ndim == 1:
            # a single point
            return self.concentration
        elif points.ndim == 2:
            # many points    
            return np.full(len(points), self.concentration)
        else:
            raise ValueError('Expected single point of list of points')


    def add_amount(self, point: np.ndarray, amount: float):
        """ add the given amount to the background
        
        Args:
            point:
                Not used
            amount:
                The total amount added to the background
        """
        self.data[0] += amount / self.volume
        
    
    def make_get_concentration_compiled(self) -> Callable:
        """ get a compiled function for obtaining concentrations
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`), which determines the concentration
            at point `point` given the background state `data`.
        """
        @nb.jit
        def get_concentration(data: np.ndarray, point: np.ndarray):
            return data[0]
        return get_concentration  # type: ignore
    
        
    def make_add_amount_compiled(self) -> Callable:
        """ get a compiled function for adding amount to the background
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`, amount: float), which adds `amount`
            to the background state given by `data` at point `point`.
        """
        volume = self.volume
     
        @nb.jit
        def add_amount(data: np.ndarray, point: np.ndarray, amount: float):        
            data += amount / volume
            
        return add_amount  # type: ignore
    




class ScalarFieldElement(FieldElementBase):
    """ the state associated with a spatially resolved background """


    parameters_default = [
        Parameter('label', '', str),
        Parameter('grid', None,
                  description='The grid defining the background field',
                  extra={'serializer': lambda grid: grid.state_serialized,
                         'unserializer': lambda state: GridBase.from_state(state)})
    ]


    def __init__(self, data: float = 0,
                 parameters: Dict[str, Any] = None):
        """ 
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            data (:class:`numpy.ndarray` or float, optional):
                Field values at the support points of the grid
        """
        super().__init__(data, parameters)
        
        if not isinstance(self.grid, CartesianGridBase):
            raise NotImplementedError('The simulations are only been '
                                      'implemented for Cartesian grids and not '
                                      f'for {self.grid.__class__.__name__}')
        
        self._field = ScalarField(self.grid, data, label=self.parameters['label'])
        self.data = self._field.data
        self.set_bounds(self.grid.axes_bounds)

        
    @classmethod
    def from_field(cls, field: ScalarField):
        """ create a scalar background state from a scalar field
        
        Args:
            field (:class:`~pde.fields.scalar.ScalarField`):
                The scalar field that initializes the background
        
        Returns:
            :class:`ScalarFieldElement`: The initialized instance
        """
        return cls(field.data, {'grid': field.grid, 'label': field.label})
        

    @property
    def grid(self) -> CartesianGridBase:
        """ :class:`~pde.grids.cartesian.CartesianGridBase`: the grid """
        return self.parameters['grid']  # type: ignore
    
    
    @property
    def field(self) -> ScalarField:
        """ :class:`~pde.fields.scalar.ScalarField`: the scalar field """
        return self._field

            
    def plot(self, ax=None, **kwargs):
        """ plot the background as a scalar field
        
        This simply calls :meth:`~pde.fields.base.DataFieldBase.plot` and all
        arguments are forwarded.
        """
        return self._field.plot(ax=ax, **kwargs)


    @classmethod
    def from_hdf_dataset(cls, dataset) -> "ScalarFieldElement":
        """ construct the state by reading data from an hdf5 dataset """
        # copy attributes from hdf
        attributes = dict(dataset.attrs)
        attributes.pop('class', None)  # remove 'class' if it is present
        
        # unserialize the attributes
        attributes = cls.unserialize_attributes(attributes)
        return cls(data=dataset, parameters=attributes['parameters'])


    @property
    def total_amount(self) -> float:
        """ float: the total material amount in the background """
        return self._field.integral


    def get_concentration(self, points: np.ndarray):
        """ determine concentration at the given points
        
        Args:
            points (:class:`numpy.ndarray`):
                The coordinates of the single point or the list of points at
                which the concentration is returned
        """
        return self._field.interpolate(points)


    def add_amount(self, point: np.ndarray, amount: float):
        """ add the given amount to the background
        
        Args:
            point (:class:`numpy.ndarray`):
                Point where the amount is added to the background
            amount (float):
                The total amount added to the background
        """
        self._field.add_interpolated(point, amount)

    
    def make_get_concentration_compiled(self) -> Callable:
        """ get a compiled function for obtaining concentrations
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`), which determines the concentration
            at point `point` given the background state `data`.
        """
        return self._field.grid.make_interpolator_compiled()


    def make_add_amount_compiled(self) -> Callable:
        """ get a compiled function for adding amount to the background
        
        Returns:
            callable: a function with signature (data: :class:`numpy.ndarray`,
            point: :class:`numpy.ndarray`, amount: float), which adds `amount`
            to the background state given by `data` at point `point`.
        """
        return self._field.grid.make_add_interpolated_compiled()

