"""
Module defining the abstract base class of elements

Elements combine the functionality of :class:`~modelrunner.parameters.Parameterized`,
which allows them to define inheritable parameters and sub-classes of
:class:`~modelrunner.state.StateBase`, which deals with input and output.
There are state classes that represent data in a form of python object, a
single numpy array, and a collection of numpy arrays:

.. autosummary::
   :nosignatures:

   ArrayElementBase
   ArrayCollectionElementBase
   ObjectElementBase

The inheritance diagram reads

.. inheritance-diagram:: ObjectElementBase ArrayElementBase ArrayCollectionElementBase
   :parts: 1
   :private-bases:

.. autoclass::
   _ElementBase

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
import warnings
from abc import ABCMeta, abstractproperty
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numba.typed import Dict as NumbaDict

from modelrunner.parameters import Parameter, Parameterized
from modelrunner.state import (
    ArrayCollectionState,
    ArrayState,
    NoData,
    ObjectState,
    StateBase,
    simplify_data,
)

SerializedAttributesType = Dict[str, str]
SerializedDataType = Union[np.ndarray, Dict[str, np.ndarray]]

if TYPE_CHECKING:
    from ..actors.base import ActorBase


class _ElementBase(Parameterized, StateBase, metaclass=ABCMeta):
    """(private) base class for representing simulation element

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

    parameters_default = [
        Parameter("plot_args", {}, dict, "Extra arguments for plotting this element")
    ]

    dim: Optional[int]  # dimensionality of the space in which the element is embedded

    _subclasses: Dict[str, _ElementBase] = {}  # type: ignore
    _compatible_actors: Sequence[ActorBase] = []

    data: Any  # defines the python access point

    _state_attributes_attr_name = "attributes"
    _state_data_attr_name = "data"

    def __init__(self, data, parameters: Optional[Dict[str, Any]] = None):
        self._state_init({"parameters": parameters}, data)

    def _state_init(self, attributes: Dict[str, Any], data=NoData) -> None:
        """initialize the state with attributes and (optionally) data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system
        """
        # set the parameters
        parameters = attributes.pop("parameters", None)
        self.parameters = self._parse_parameters(
            parameters, include_deprecated=True, check_validity=True
        )

        # initialize the attributes and data of StateBase
        super()._state_init(attributes, data)

        if attributes:
            raise ValueError(f"Too many attributes: {attributes.keys()}")

    @property
    def _data_numba(self):
        return self.data

    @property
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {"parameters": self.parameters}

    @property
    def _state_attributes_store(self) -> Dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage

        This property modifies the normal `_state_attributes` and adds information
        necessary for restoring the class using :meth:`StateBase.from_data`.
        """
        attrs = super()._state_attributes_store

        if "parameters" in attrs:
            # serialize the individual parameters
            default_parameters = self.get_parameters(
                include_hidden=True, include_deprecated=True, sort=False
            )

            for key, value in attrs["parameters"].items():
                if key in default_parameters:
                    def_param_extra = default_parameters[key].extra
                    if "serializer" in def_param_extra:
                        attrs["parameters"][key] = def_param_extra["serializer"](value)
                        continue
                attrs["parameters"][key] = simplify_data(value)

        return attrs

    @classmethod
    def _unpack_parameters(cls, parameters: Dict[str, Any]) -> None:
        """convert an attribute from a form that was stored"""
        default_parameters = cls.get_parameters(
            include_hidden=True, include_deprecated=True, sort=False
        )

        # unserialize the individual parameters
        for key in parameters:
            if key in default_parameters:
                def_param_extra = default_parameters[key].extra
                if "unserializer" in def_param_extra:
                    parameters[key] = def_param_extra["unserializer"](parameters[key])

    @classmethod
    def from_data(cls, attributes: Dict[str, Any], data=NoData) -> _ElementBase:
        """create instance of any state class from attributes and data

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degerees of freedom of the physical system

        Returns:
            The object containing the given attributes and data
        """
        cls._unpack_parameters(attributes["parameters"])
        return super().from_data(attributes=attributes, data=data)

    def __setstate__(self, dictdata):
        """set all properties of the object from a stored representation"""
        self._unpack_parameters(dictdata["attributes"]["parameters"])
        super().__setstate__(dictdata)

    def __str__(self):
        return f"{self.__class__.__name__}(...)"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.data}, parameters={self.parameters})"
        )

    @abstractproperty
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        ...

    def plot(self, ax=None, *args, **kwargs):
        """plot the element"""
        pass

    def _get_napari_layer_data(self, **kwargs) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

        Returns:
            dict: all the information necessary to plot this element
        """
        raise NotImplementedError


class ObjectElementBase(_ElementBase, ObjectState):
    """Element storing data in a python object"""

    def __init__(self, data, parameters: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        ObjectState.__init__(self, data)
        self._state_init({"parameters": parameters})

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        try:
            return len(self.data)
        except AttributeError:
            return 1


class ArrayElementBase(_ElementBase, ArrayState):
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
        self._state_init({"parameters": parameters}, data)

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


class ArrayCollectionElementBase(_ElementBase, ArrayCollectionState):
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
        ArrayCollectionState.__init__(self, data)
        self._state_init({"parameters": parameters})

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
# class DictElementBase(_ElementBase, DictState):
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
