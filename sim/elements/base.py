"""
Module defining the abstract base class of elements

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABCMeta, abstractproperty
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
from numba.typed import Dict as NumbaDict

from modelrunner.parameters import Parameterized
from modelrunner.state import ArrayState, ObjectState
from pde.tools.cache import objects_equal

SerializedAttributesType = Dict[str, str]
SerializedDataType = Union[np.ndarray, Dict[str, np.ndarray]]

if TYPE_CHECKING:
    from ..actors.base import ActorBase


class ElementBase(Parameterized, metaclass=ABCMeta):
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

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None) -> ElementBase:
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
    def attributes(self) -> Dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        attributes = super().attributes
        attributes["parameters"] = self.parameters
        return attributes

    #
    # def serialize_attribute(self, name: str, value) -> str:
    #     """serialize an attribute into a string
    #
    #     Args:
    #         name (str): Name of the attribute
    #         value: The value of the attribute that needs to be serialized
    #
    #     Returns:
    #         str: A string representation from which the `value` can be reconstructed
    #     """
    #     if name == "parameters":
    #         # serialize the individual parameters
    #         default_parameters = self.get_parameters(
    #             include_hidden=True, include_deprecated=True, sort=False
    #         )
    #
    #         parameters = {}
    #         for key in self.parameters:
    #             serializer = json.dumps
    #             if key in default_parameters:
    #                 def_param_extra = default_parameters[key].extra
    #                 if "serializer" in def_param_extra:
    #                     serializer = def_param_extra["serializer"]
    #             parameters[key] = serializer(value[key])
    #         value = parameters
    #
    #     # serialize the value using JSON
    #     try:
    #         return json.dumps(value)
    #     except TypeError as e:
    #         msg = f'Cannot serialize "{key}" of "{self.__class__.__name__}"'
    #         raise TypeError(msg) from e
    #
    # @classmethod
    # def unserialize_attribute(cls, name: str, value_str: str) -> Any:
    #     """unserializes the given attribute
    #
    #     Args:
    #         name (str): Name of the attribute
    #         value_str (str): Serialized value of the attribute
    #
    #     Returns:
    #         The unserialized value
    #     """
    #     # unserialize assuming it is JSON-encoded
    #     value = json.loads(value_str)
    #
    #     if name == "parameters":
    #         # unserialize the individual parameters
    #         default_parameters = cls.get_parameters(
    #             include_hidden=True, include_deprecated=True, sort=False
    #         )
    #
    #         for key in value:
    #             unserializer = json.loads
    #             if key in default_parameters:
    #                 def_param_extra = default_parameters[key].extra
    #                 if "unserializer" in def_param_extra:
    #                     unserializer = def_param_extra["unserializer"]
    #             value[key] = unserializer(value[key])
    #
    #     return value
    #
    # def _write_hdf(self, root, key: str = "data", **kwargs):
    #     r"""store element state in a file
    #
    #     Args:
    #         filename (str):
    #             Path where the data is stored
    #         \**kwargs:
    #             Additional parameters may be supported for some formats
    #     """
    #     dataset = root.create_dataset(key, data=self.data)
    #
    #     # write serialized attributes
    #     for name, value in self.attributes.items():
    #         dataset.attrs[name] = self.serialize_attribute(name, value)
    #
    # def __eq__(self, other):
    #     if not isinstance(other, self.__class__):
    #         return NotImplemented
    #     return self.attributes == other.attributes and objects_equal(
    #         self.data, other.data
    #     )

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
        ElementBase.__init__(self, parameters)
        ObjectState.__init__(self, data)

    @property
    def _data_numba(self) -> Any:
        """returns the data associated with the state in a form that numba can handle"""
        return self.data


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
        ElementBase.__init__(self, parameters)
        if data is not None:
            ArrayState.__init__(self, data)
            self._data_numba = self.data
