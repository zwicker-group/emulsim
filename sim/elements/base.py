"""
Module defining the abstract base class of elements

States combine the functionality of :class:`Parameterized`, which allows them to define
inheritable parameters and sub-classes of :class:`StateBase`, which deals with input and
output. There are state classes that represent data in a form of python object, a
single numpy array, and a collection of numpy arrays:

.. inheritance-diagram:: ObjectElementBase ArrayElementBase ArrayCollectionElementBase
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import math
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numba.typed import Dict as NumbaDict

from modelrunner.io import simplify_data
from modelrunner.parameters import Parameterized
from modelrunner.state import ArrayCollectionState, ArrayState, ObjectState, StateBase

SerializedAttributesType = Dict[str, str]
SerializedDataType = Union[np.ndarray, Dict[str, np.ndarray]]

if TYPE_CHECKING:
    from ..actors.base import ActorBase


class ElementBase(Parameterized, StateBase, metaclass=ABCMeta):
    """represents a simulation element

    Elements are generally characterized by a `data` attribute, which contains
    information about all degrees of freedom, and `parameters`, which contain additional
    information in form of a python dictionary. While the parameters are managed by the
    mixin :class:`~modelrunner.parameters.Parameterized`, the form of the data depends
    on the element and must thus be defined by concrete classes. These classes need to
    define at least to access points into the data: An attribute `data`, which is the
    main access point for normal python code, and an attribute `_data_numba`, which
    is used by `numba` to access and alter the underlying data. In many cases, these two
    attributes can point to the same object, e.g., a :class:`~numpy.ndarray`.
    """

    dim: Optional[int]  # dimensionality of the space in which the element is embedded

    _subclasses: Dict[str, ElementBase] = {}  # type: ignore
    _compatible_actors: Sequence[ActorBase] = []

    data: Any  # defines the python access point
    _data_numba: Any  # defines the numba access point

    @abstractmethod
    def __init__(self, data, parameters: Optional[Dict[str, Any]] = None):
        ...

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=None) -> ElementBase:
        """create the element state from attributes and data

        Args:
            attributes (dict):
                Attributes of the element. This carries information about
                parameters and possibly additional parts that do not depend on
                time.
            data (:class:`~numpy.ndarray`):
                The numerical data associated with the state of the element
        """
        if "__class__" in attributes and attributes["__class__"] != cls.__name__:
            logger = logging.getLogger(__name__)
            logger.warning(
                f'Initialize `{cls.__name__}` with data from `{attributes["class"]}`'
            )
        return cls(data, attributes.get("parameters", None))

    def __str__(self):
        return f"{self.__class__.__name__}(...)"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.data}, parameters={self.parameters})"
        )

    @property
    def _data_numba(self):
        return self.data

    @abstractproperty
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        ...

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        attributes = super().attributes
        attributes["parameters"] = self.parameters
        return attributes

    def _pack_attribute(self, name: str, value) -> Any:
        """convert an attribute into a form that can be stored

        If this function raises :class:`DoNotStore`, the attribute will not be stored

        Args:
            name (str): Name of the attribute
            value: The value of the attribute

        Returns:
            A simplified form of the attribute that can be restored
        """
        if name == "parameters":
            # serialize the individual parameters
            default_parameters = self.get_parameters(
                include_hidden=True, include_deprecated=True, sort=False
            )

            parameters = {}
            for key in self.parameters:
                if key in default_parameters:
                    def_param_extra = default_parameters[key].extra
                    if "serializer" in def_param_extra:
                        parameters[key] = def_param_extra["serializer"](value[key])
                        continue
                parameters[key] = simplify_data(value[key])
            return parameters

        return super()._pack_attribute(name, value)

    @classmethod
    def _unpack_attribute(cls, name: str, value: Any) -> Any:
        """convert an attribute from a form that was stored

        Args:
            name (str): Name of the attribute
            value: The value of the attribute

        Returns:
            A restored form of the attribute
        """
        if name == "parameters":
            # unserialize the individual parameters
            default_parameters = cls.get_parameters(
                include_hidden=True, include_deprecated=True, sort=False
            )

            for key in value:
                if key in default_parameters:
                    def_param_extra = default_parameters[key].extra
                    if "unserializer" in def_param_extra:
                        value[key] = def_param_extra["unserializer"](value[key])
            return value

        return super()._unpack_attribute(name, value)

    def plot(self, ax=None, *args, **kwargs):
        """plot the element"""
        pass

    def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

        Returns:
            dict: all the information necessary to plot this element
        """
        raise NotImplementedError


class ObjectElementBase(ElementBase, ObjectState):
    """Element storing data in a python object"""

    def __init__(self, data, parameters: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        Parameterized.__init__(self, parameters)
        ObjectState.__init__(self, data)

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        try:
            return len(self.data)
        except AttributeError:
            return 1


class ArrayElementBase(ElementBase, ArrayState):
    """Element storing data in a numpy array"""

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        Parameterized.__init__(self, parameters)
        if data is not None:
            ArrayState.__init__(self, data)

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        arr = np.asanyarray(self.data)
        if arr.dtype.fields:
            # array is a structured array or record array with fields
            itemsize = sum(
                math.prod(fields[0].shape) for fields in arr.dtype.fields.values()
            )
        else:
            # array is a simple array
            itemsize = 1
        return int(arr.size * itemsize)


class ArrayCollectionElementBase(ElementBase, ArrayCollectionState):
    """Element storing data in multiple numpy array"""

    def __init__(
        self,
        data: Optional[Tuple[np.ndarray, ...]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        Parameterized.__init__(self, parameters)
        ArrayCollectionState.__init__(self, data)

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        dof = 0
        for arr in self.data:
            if arr.dtype.fields:
                # array is a structured array or record array with fields
                itemsize = sum(
                    math.prod(fields[0].shape) for fields in arr.dtype.fields.values()
                )
            else:
                # array is a simple array
                itemsize = 1
            dof += int(arr.size * itemsize)
        return dof


#
# class DictElementBase(ElementBase, DictState):
#     """Element storing data in a dictionary of states"""
#
#     def __init__(
#         self,
#         data: Optional[Dict[str, StateBase]] = None,
#         parameters: Optional[Dict[str, Any]] = None,
#     ):
#         """
#         Args:
#             data: The data describing the state
#             parameters: Additional parameters that affect the element
#         """
#         Parameterized.__init__(self, parameters)
#         DictState.__init__(self, data)
#
#     @property
#     def _data_numba(self) -> Tuple:
#         """returns the data associated with the state in a form that numba can handle"""
#         return tuple(state._data_numba for state in self.data.values())
#
#     @property
#     def degrees_of_freedom(self) -> int:
#         """int: the number of degrees of freedom for this element"""
#
#         arr = np.asanyarray(self.data)
#         if arr.dtype.fields:
#             # array is a structured array or record array with fields
#             itemsize = sum(
#                 math.prod(fields[0].shape) for fields in arr.dtype.fields.values()
#             )
#         else:
#             # array is a simple array
#             itemsize = 1
#         return int(arr.size * itemsize)
