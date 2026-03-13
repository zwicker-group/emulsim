"""Module defining the abstract base class of elements.

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

import copy
import math
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
from numba import literal_unroll

from modelrunner import Parameter, Parameterized
from modelrunner.storage import (
    Attrs,
    Location,
    ModeType,
    StorageGroup,
    StorageID,
    open_storage,
)
from modelrunner.storage.utils import decode_class, storage_actions
from pde.backends.numba.utils import jit

SerializedAttributesType = dict[str, str]
SerializedDataType = np.ndarray | dict[str, np.ndarray]

if TYPE_CHECKING:
    from ..actors.base import ActorBase


class NoData:
    """Helper class that marks data omission."""


def _equals(left: Any, right: Any) -> bool:
    """Checks whether two objects are equal, also supporting :class:~numpy.ndarray`

    Args:
        left: one object
        right: other object

    Returns:
        bool: Whether the two objects are equal
    """
    if type(left) is not type(right):
        return False

    if isinstance(left, str):
        return bool(left == right)

    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)

    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equals(left[key], right[key]) for key in left
        )

    if hasattr(left, "__iter__"):
        return len(left) == len(right) and all(
            _equals(l, r) for l, r in zip(left, right, strict=False)
        )

    return bool(left == right)


TElement = TypeVar("TElement", bound="_ElementBase")


class _ElementBase(Parameterized, metaclass=ABCMeta):
    """(private) base class for representing simulation element.

    Elements are generally characterized by a `data` attribute, which contains
    information about all degrees of freedom, and `parameters`, which contain additional
    information in form of a python dictionary. While the parameters are managed by the
    mixin :class:`~modelrunner.model.parameters.Parameterized`, the form of the data
    depends on the element and must thus be defined by concrete classes. These classes
    need to define at least to access points into the data: An attribute `data`, which
    is the main access point for normal python code, and an attribute `_data_numba`,
    which is used by `numba` to access and alter the underlying data. In many cases,
    these two attributes can point to the same object, e.g., a :class:`~numpy.ndarray`.
    """

    parameters_default = [
        Parameter("plot_args", {}, dict, "Extra arguments for plotting this element")
    ]

    dim: int | None  # dimensionality of the space in which the element is embedded

    _element_types: dict[str, type[_ElementBase]] = {}
    _compatible_actors: Sequence[ActorBase] = []
    _format_version: int = 1

    data: Any  # defines the python access point

    def __init__(self, data: Any, parameters: dict[str, Any] | None = None):
        Parameterized.__init__(self, parameters, strict=True)
        self._init_state({"parameters": parameters}, data)

    def __init_subclass__(cls, **kwargs):
        """Register all subclasses to reconstruct them later."""
        # register the subclasses
        super().__init_subclass__(**kwargs)
        if cls is not _ElementBase:
            if cls.__name__ in _ElementBase._element_types:
                warnings.warn(f"Redefining class {cls.__name__}")
            _ElementBase._element_types[cls.__name__] = cls
            storage_actions.register("read_item", cls, cls._from_stored_data)
            storage_actions.register("write_item", cls, cls._write_to_storage)

    def _init_state(self, attributes: dict[str, Any], data=NoData) -> None:
        """Initialize the state from attributes and (optionally) data.

        This function is the central intialization method for the element, which is
        called by :meth:`__init__`, :meth:`__setstate__`, and :meth:`from_data`.

        Args:
            attributes (dict):
                Additional (unserialized) attributes
            data:
                The data of the degrees of freedom of the physical system
        """
        # set the parameters
        parameters = attributes.pop("parameters", None)
        self.parameters = self._parse_parameters(
            parameters, include_deprecated=True, check_validity=True
        )

        # raise exception if there are unused attributes
        if attributes:
            raise ValueError(f"Too many attributes: {attributes.keys()}")

        # set the data
        if data is not NoData:
            self.data = data

    @property
    def _data_numba(self):
        return self.data

    @property
    def attributes(self) -> dict[str, Any]:
        """dict: information about the element state, which does not change in time"""
        return {"parameters": self.parameters}

    @property
    def _attributes_storage(self) -> dict[str, Any]:
        """dict: Attributes in the form in which they will be written to storage

        This property modifies the normal `attributes` and adds information
        necessary for restoring the class using :meth:`from_data`.
        """
        # make a copy since we add additional fields below
        attrs = copy.deepcopy(self.attributes)

        # add some additional information
        attrs["_element_class"] = self.__class__.__name__
        attrs["_format_version"] = self._format_version

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
                attrs["parameters"][key] = value

        return attrs

    @classmethod
    def _unpack_parameters(cls, parameters: dict[str, Any]) -> None:
        """Convert an attribute from a form that was stored."""
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
    def from_data(cls, attributes: dict[str, Any], data=NoData) -> _ElementBase:
        """Create instance of any state class from attributes and data.

        Args:
            attributes (dict): Additional (unserialized) attributes
            data: The data of the degrees of freedom of the physical system

        Returns:
            The object containing the given attributes and data
        """
        cls._unpack_parameters(attributes["parameters"])

        # copy attributes since they are modified in this function
        attributes = attributes.copy()
        cls_name = attributes.pop("_element_class", None)

        if cls_name is None or cls.__name__ == cls_name:
            # attributes contain right class name or no class information at all
            # => instantiate current class with given data
            format_version = attributes.pop("_format_version", 0)
            if format_version != cls._format_version:
                warnings.warn(
                    f"File format version mismatch "
                    f"({format_version} != {cls._format_version})"
                )

            # create a new object without calling __init__, which might be overwriten by
            # the subclass and not follow our interface
            obj = cls.__new__(cls)
            obj._init_state(attributes, data)
            return obj

        elif cls is _ElementBase:
            # use the base class as a point to load arbitrary subclasses
            if cls_name == "_ElementBase":
                raise RuntimeError("Cannot create _ElementBase instances")
            state_cls = cls._element_types[cls_name]
            return state_cls.from_data(attributes, data)

        else:
            raise ValueError(f"Incompatible state class {cls_name}")

    def __getstate__(self) -> dict[str, Any]:
        """Return a representation of the current state.

        Note that this representation might contain views into actual data
        """
        attrs = self._attributes_storage
        # remove private attributes used for persistent storage
        attrs.pop("_element_class")
        attrs.pop("_format_version")
        return {"attributes": attrs, "data": self.data}

    def __setstate__(self, dictdata):
        """Set all properties of the object from a stored representation."""
        self._unpack_parameters(dictdata["attributes"]["parameters"])
        self._init_state(dictdata.get("attributes", {}), dictdata.get("data", NoData))

    def copy(
        self: TElement, method: Literal["clean", "shallow", "data"], data=None
    ) -> TElement:
        """Create a copy of the state.

        There are several methods of copying the state:

        `clean`:
            Makes a copy of the state by gathering its contents using
            :meth:`~StateBase.__getstate__`, making a copy of only the actual data and
            then instantiating a new state class, using :meth:`~StateBase.__setstate__`
            to restore the state. Since a new object is created, all data not captured
            by `__getstate__` (like internal caches) are lost!
        `shallow`:
            Performs a shallow copy of all attributes of the class. This is simply
            copying the entire :attr:`__dict__`
        `data`:
            Like `shallow`, but additionally makes a deep copy of the state data stored
            in the :attr:`data`.

        Args:
            method (str):
                Determines whether a `clean`, `shallow`, or `data` copy is performed.
                See description above for details.
            data:
                Data to be used instead of the one in the current state. This data is
                used as is and not copied!

        Returns:
            A copy of the current state object
        """
        # create a new object of the same class without any attributes
        obj = self.__class__.__new__(self.__class__)

        if method == "clean":
            # make clean copy by re-initializing state with copy of relevant attributes
            state = copy.deepcopy(self.__getstate__())  # copy current state
            if data is not None:
                state["data"] = data
            # use __setstate__ to set data on new object
            obj.__setstate__(state)

        elif method == "shallow":
            # (shallow) copy of all attributes of current state, including `data`
            obj.__dict__ = self.__dict__.copy()
            if data is not None:
                obj.data = data

        elif method == "data":
            # (shallow) copy of all attributes of current state, except `data`, which is
            # copied using a deep-copy
            obj.__dict__ = self.__dict__.copy()
            if data is None:
                if isinstance(self, DictElementBase):
                    # special implementation for copying elements in a dictionary
                    obj.data = {k: v.copy(method="data") for k, v in self.data.items()}
                else:
                    obj.data = copy.deepcopy(self.data)
            else:
                obj.data = data

        else:
            raise ValueError(f"Unknown copy method {method}")
        return obj

    def __str__(self):
        return f"{self.__class__.__name__}(...)"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.data}, parameters={self.parameters})"
        )

    def __eq__(self, other) -> bool:
        if self.__class__ is not other.__class__:
            return False
        if self.attributes != other.attributes:
            return False
        return _equals(self.data, other.data)

    @property
    @abstractmethod
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        ...

    def _make_error_estimator(self, backend: str) -> Callable[[Any, Any], float]:
        """Return function that estimates the error between element data."""
        raise NotImplementedError("Element does not implement error estimator")

    @classmethod
    def _get_attrs_from_storage(
        cls, storage: StorageGroup, loc: Location, *, check_version: bool = True
    ) -> Attrs:
        """Read attributes from storage and optionally check format version.

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                Name of the location where the data will be stored.
            check_version (bool):
                A number indicating whether the format version should be checked

        Raises:
            `RuntimeError`: If format version is specified, but not matched

        Returns:
            dict: Attributes without the `_element_class` and `_format_version` item
        """
        # read relevant attributes of the state
        attrs = storage.read_attrs(loc)

        # check whether the data can be read
        attrs.pop("_element_class", None)
        version = attrs.pop("_format_version", None)
        if check_version is not None and version != cls._format_version:
            raise RuntimeError(f"Cannot read format version {version}")

        if "parameters" in attrs:
            cls._unpack_parameters(attrs["parameters"])

        return attrs

    @classmethod
    def _from_stored_data(
        cls, storage: StorageGroup, loc: Location, *, index: int | None = None
    ):
        """Create the element from some storage.

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        # determine the class to reconstruct the data from attribute

        class_name = storage.read_attrs(loc).get("_element_class", None)
        if class_name in _ElementBase._element_types:
            element_cls = _ElementBase._element_types[class_name]
        else:
            element_cls = decode_class(class_name)  # type: ignore
            if element_cls is None:
                raise RuntimeError(f"Could not decode class `{class_name}`")

        if element_cls == _ElementBase:
            raise NotImplementedError(f"Cannot read `{cls.__name__}`")
        else:
            return element_cls._from_stored_data(storage, loc, index=index)

    def _update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: int | None = None
    ) -> None:
        """Update the state data (but not its attributes) from storage.

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is read
            index (int, optional):
                If the location contains a trajectory of the state, `index` must denote
                the index determining which state should be created
        """
        raise NotImplementedError(f"Cannot update `{self.__class__.__name__}`")

    def _write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        """Write the state to storage.

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the state is written
        """
        raise NotImplementedError(f"Cannot write `{self.__class__.__name__}`")

    def _create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """Prepare a trajectory of the current state.

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        raise NotImplementedError(
            f"Cannot create trajectory for `{self.__class__.__name__}`"
        )

    def _append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        """Append the current state to a prepared trajectory.

        Args:
            storage (:class:`StorageGroup`):
                A storage opened with :func:`~modelrunner.storage.open_storage`
            loc (str or list of str):
                The location in the storage where the trajectory is written
        """
        raise NotImplementedError(
            f"Cannot extend trajectory for `{self.__class__.__name__}`"
        )

    @classmethod
    def from_file(cls, storage: StorageID, loc: Location = "state", **kwargs):
        r"""Load object from a file.

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                Path or instance describing the storage. The simplest choice is a path
                to a file, where the data is written in a format deduced from the file
                extension.
            loc (str or list of str):
                Name of the location where the data was stored.
            **kwargs:
                Arguments passed to :func:`~modelrunner.storage.open_storage`
        """
        kwargs.setdefault("mode", "read")
        with open_storage(storage, **kwargs) as opened_storage:
            return _ElementBase._from_stored_data(opened_storage, loc)

    def to_file(
        self,
        storage: StorageID,
        loc: Location = "state",
        *,
        mode: ModeType = "insert",
        **kwargs,
    ) -> None:
        """Write this object to a file.

        Args:
            storage (str or :class:`~modelrunner.storage.StorageBase`):
                Path or instance describing the storage. The simplest choice is a path
                to a file, where the data is written in a format deduced from the file
                extension.
            loc (str or list of str):
                Name of the location where the data will be stored.
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed, which determines the
                allowed operations. Common options are "read", "full", "append", and
                "truncate".
            **kwargs:
                Arguments passed to :func:`~modelrunner.storage.open_storage`
        """
        with open_storage(storage, mode=mode, **kwargs) as opened_storage:
            self._write_to_storage(opened_storage, loc=loc)

    def plot(self, ax=None, *args, **kwargs):
        """Plot the element."""

    def _get_napari_layer_data(self, **kwargs) -> dict[str, Any]:
        """Returns data for plotting on a single napari layer.

        Returns:
            dict: all the information necessary to plot this element
        """
        raise NotImplementedError


class ObjectElementBase(_ElementBase):
    """Element storing data in a python object."""

    @property
    def degrees_of_freedom(self) -> int:
        """int: the number of degrees of freedom for this element"""
        try:
            return len(self.data)
        except AttributeError:
            return 1

    @classmethod
    def _from_stored_data(
        cls, storage: StorageGroup, loc: Location, *, index: int | None = None
    ) -> ObjectElementBase:
        attrs = cls._get_attrs_from_storage(storage, loc, check_version=True)

        # create the state from the read data
        data = storage.read_array(loc, index=index).item()
        obj = cls.__new__(cls)
        obj._init_state(attrs, data)
        return obj

    def _update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: int | None = None
    ) -> None:
        storage.read_array(loc, index=index, out=self.data)

    def _write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        arr: np.ndarray = np.empty((), dtype=object)
        arr[()] = self.data
        storage.write_array(
            loc, arr, attrs=self._attributes_storage, cls=self.__class__
        )

    def _create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        storage.create_dynamic_array(
            loc,
            shape=(),
            dtype=object,
            attrs=self._attributes_storage,
            cls=self.__class__,
        )

    def _append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        arr: np.ndarray = np.empty((), dtype=object)
        arr[()] = self.data
        storage.extend_dynamic_array(loc, arr)


class ArrayElementBase(_ElementBase):
    """Element storing data in a numpy array."""

    def __init__(
        self,
        data: np.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        super().__init__(data, parameters)

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

    def _make_error_estimator(self, backend: str) -> Callable[[Any, Any], float]:
        """Return function that estimates the error between element data.

        Args:
            backend (str): The backend used to calculate the error
        """
        arr = np.asanyarray(self._data_numba)

        if backend == "numpy":

            def error_estimator(data1: np.ndarray, data2: np.ndarray) -> float:
                # we view the data as floats to also deal with structured arrays
                return np.abs(data1.view(float) - data2.view(float)).max()  # type: ignore

        elif backend == "numba":
            if arr.dtype.fields:
                # iterate over all fields of this structured array
                field_names = tuple(arr.dtype.fields.keys())

                @jit
                def error_estimator(data1: np.ndarray, data2: np.ndarray) -> float:
                    error = 0
                    for field in literal_unroll(field_names):
                        error = max(np.abs(data1[field] - data2[field]).max(), error)
                    return error

            else:

                @jit
                def error_estimator(data1: np.ndarray, data2: np.ndarray) -> float:
                    return np.abs(data1 - data2).max()  # type: ignore

        else:
            raise NotImplementedError(f"Unknown backend `{backend}`")

        return error_estimator

    @classmethod
    def _from_stored_data(
        cls, storage: StorageGroup, loc: Location, *, index: int | None = None
    ) -> ArrayElementBase:
        attrs = cls._get_attrs_from_storage(storage, loc, check_version=True)

        # create the state from the read data
        data = storage.read_array(loc, index=index)
        obj = cls.__new__(cls)
        obj._init_state(attrs, data)
        return obj

    def _update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: int | None = None
    ) -> None:
        storage.read_array(loc, index=index, out=self.data)

    def _write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        storage.write_array(
            loc, self.data, attrs=self._attributes_storage, cls=self.__class__
        )

    def _create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        storage.create_dynamic_array(
            loc,
            shape=self.data.shape,
            dtype=self.data.dtype,
            attrs=self._attributes_storage,
            cls=self.__class__,
        )

    def _append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        storage.extend_dynamic_array(loc, self.data)


class ArrayCollectionElementBase(_ElementBase):
    """Element storing data in multiple numpy array."""

    def __init__(
        self,
        data: tuple[np.ndarray, ...] | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        super().__init__(data, parameters)

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

    def _make_error_estimator(self, backend: str) -> Callable[[Any, Any], float]:
        """Return function that estimates the error between element data.

        Args:
            backend (str): The backend used to calculate the error
        """

        def error_estimator(data1: np.ndarray, data2: np.ndarray) -> float:
            error = 0
            for arr1, arr2 in zip(data1, data2, strict=False):
                error = max(np.abs(arr1 - arr2).max())
            return error

        if backend == "numba":
            return jit(error_estimator)  # type: ignore
        else:
            return error_estimator


class DictElementBase(_ElementBase):
    """Element storing data in a dictionary of states."""

    def __init__(
        self,
        data: dict[str, _ElementBase] | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """
        Args:
            data: The data describing the state
            parameters: Additional parameters that affect the element
        """
        super().__init__(data, parameters)

    @property
    def _attributes_storage(self) -> dict[str, Any]:
        attrs = super()._attributes_storage
        attrs["_element_labels"] = list(self.data.keys())
        return attrs

    @property
    def _data_numba(self) -> tuple:
        """Returns the data associated with the state in a form that numba can
        handle."""
        return tuple(state._data_numba for state in self.data.values())

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

    @classmethod
    def _from_stored_data(
        cls, storage: StorageGroup, loc: Location, *, index: int | None = None
    ) -> DictElementBase:
        attrs = cls._get_attrs_from_storage(storage, loc, check_version=True)

        # read the data
        subgroup = storage.open_group(loc)
        data = {
            label: _ElementBase._from_stored_data(subgroup, label, index=index)
            for label in attrs["_element_labels"]
        }

        # create the state from the read data
        obj = cls.__new__(cls)
        obj._init_state(attrs, data)
        return obj

    def _update_from_stored_data(
        self, storage: StorageGroup, loc: Location, index: int | None = None
    ) -> None:
        subgroup = storage.open_group(loc)
        for key, substate in self.data.items():
            substate._update_from_stored_data(subgroup, key, index=index)

        storage.read_array(loc, index=index, out=self.data)

    def _write_to_storage(self, storage: StorageGroup, loc: Location) -> None:
        subgroup = storage.create_group(
            loc, attrs=self._attributes_storage, cls=self.__class__
        )
        for label, substate in self.data.items():
            substate._write_to_storage(subgroup, label)

    def _create_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        subgroup = storage.create_group(
            loc, attrs=self._attributes_storage, cls=self.__class__
        )
        for label, substate in self.data.items():
            substate._create_trajectory(subgroup, label=label)

    def _append_to_trajectory(self, storage: StorageGroup, loc: Location) -> None:
        subgroup = storage.open_group(loc)
        for label, substate in self.data.items():
            substate._append_to_trajectory(subgroup, label)
