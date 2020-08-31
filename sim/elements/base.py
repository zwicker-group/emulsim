"""
Module defining the abstract base class of elements

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy
import json
import logging
from abc import ABCMeta
from typing import Any, Callable, Dict, Type, Union  # @UnusedImport

import numpy as np

from pde.tools.cache import objects_equal
from pde.tools.parameters import Parameterized

SerializedAttributesType = Dict[str, str]
SerializedDataType = Union[np.ndarray, Dict[str, np.ndarray]]


class ElementBase(Parameterized, metaclass=ABCMeta):
    """ represents a simulation element """

    data: np.ndarray
    """ :class:`numpy.ndarray`:
    Data describing the state of the element. These are the dynamical variables
    (degree of freedoms) of the simulation
    """

    _subclasses: Dict[str, "ElementBase"] = {}  # type: ignore

    dim: int  # dimensionality of the space in which the element is embedded

    def __init__(self, data=None, parameters: Dict[str, Any] = None):
        """
        Args:
            data: The data defining the state
            parameters (dict): Parameters affecting the behavior of the element
        """
        super().__init__(parameters)
        self.data = data

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def from_state(cls, attributes: Dict[str, Any], data=None) -> "ElementBase":
        """create the element state from attributes and data

        Args:
            attributes (dict):
                Attributes of the element. This carries information about
                parameters and possibly additional parts that do not depend on
                time.
            data (:class:`numpy.ndarray`):
                The numerical data associated with the state of the element
        """
        if "class" in attributes and attributes["class"] != cls.__name__:
            logger = logging.getLogger(__name__)
            logger.warning(
                f'Initialize `{cls.__name__}` with data from `{attributes["class"]}`'
            )
        return cls(data, attributes.get("parameters", None))

    @classmethod
    def from_hdf_dataset(cls, dataset) -> "ElementBase":
        """construct the element by reading data from an hdf5 dataset

        Args:
            dataset: the hdf5 dataset (in an already opened file)
        """
        # copy attributes from hdf
        attributes = dict(dataset.attrs)

        # determine class
        class_name = json.loads(attributes.pop("class"))
        field_cls = cls._subclasses[class_name]

        # unserialize the attributes
        attributes = cls.unserialize_attributes(attributes)

        # construct the instance
        return field_cls.from_state(attributes, data=dataset)

    def __str__(self):
        return f"{self.__class__.__name__}(...)"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.data}, parameters={self.parameters})"
        )

    @property
    def attributes(self) -> Dict[str, Any]:
        """ dict: information about the element state """
        return {"class": self.__class__.__name__, "parameters": self.parameters}

    @property
    def attributes_serialized(self) -> Dict[str, str]:
        """ dict: serialized version of the attributes """
        # serialize the individual parameters
        default_parameters = self.get_parameters(
            include_hidden=True, include_deprecated=True, sort=False
        )

        parameters = {}
        for key, value in self.parameters.items():
            serializer = json.dumps
            if key in default_parameters:
                def_param_extra = default_parameters[key].extra
                if "serializer" in def_param_extra:
                    serializer = def_param_extra["serializer"]
            parameters[key] = serializer(value)

        # serialize all remaining attributes
        attributes = self.attributes
        attributes["parameters"] = parameters

        result = {}
        for key, value in attributes.items():
            try:
                result[key] = json.dumps(value)
            except TypeError as e:
                msg = f'Cannot serialize "{key}" of "{self.__class__.__name__}"'
                raise TypeError(msg) from e
        return result

    @classmethod
    def unserialize_attributes(cls, attributes: Dict[str, str]) -> Dict[str, Any]:
        """unserializes the given attributes

        Args:
            attributes (dict):
                The serialized attributes

        Returns:
            dict: The unserialized attributes
        """
        # unserialize all attributes
        attributes = {key: json.loads(value) for key, value in attributes.items()}

        # unserialize the individual parameters
        default_parameters = cls.get_parameters(
            include_hidden=True, include_deprecated=True, sort=False
        )

        parameters = attributes["parameters"]
        for key in parameters:
            unserializer = json.loads
            if key in default_parameters:
                def_param_extra = default_parameters[key].extra
                if "unserializer" in def_param_extra:
                    unserializer = def_param_extra["unserializer"]
            parameters[key] = unserializer(parameters[key])  # type: ignore

        return attributes

    def to_file(self, filename: str, **kwargs):
        r"""store element state in a file

        Args:
            filename (str):
                Path where the data is stored
            \**kwargs:
                Additional parameters may be supported for some formats
        """
        import h5py

        with h5py.File(filename, "w") as fp:
            self._write_hdf_dataset(fp, **kwargs)

    def _write_hdf_dataset(self, hdf_path, key: str = "data"):
        """write data to a given hdf5 file pointer `hdf_path`

        Args:
            hdf_path: the hdf5 dataset (in an already opened file)
        """
        dataset = hdf_path.create_dataset(key, data=self.data)

        # write attributes
        for key, value in self.attributes_serialized.items():
            dataset.attrs[key] = value

    def copy(self, data=None):
        """create a copy of the element

        Args:
            data:
                New data to overwrite the data of the current element. If
                omitted, the data of the current element is copied.
        """
        if data is None:
            data = self.data.copy()
        attributes = copy.deepcopy(self.attributes)
        return self.__class__.from_state(attributes=attributes, data=data)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.attributes == other.attributes and objects_equal(
            self.data, other.data
        )

    @property
    def degrees_of_freedom(self) -> int:
        """ int: the number of degrees of freedom for this element """
        return int(np.asanyarray(self.data).size)

    def plot(self, ax=None, *args, **kwargs):
        """ plot the element """
        pass


def element_from_hdf(hdf_path) -> ElementBase:
    """create element instance from a stored state

    Args:
        hdf_path: HDF path in an already opened file
    """
    if "class" in hdf_path.attrs:
        # assume everything is stored in root directory
        dataset = hdf_path
    else:
        # assume a single field is stored in the data
        dataset_names = list(hdf_path.keys())
        if len(dataset_names) > 1:
            logging.getLogger(__name__).warning("Using only the first of many datasets")

        dataset = hdf_path[dataset_names[0]]  # retrieve first dataset

    # determine class
    class_name = json.loads(dataset.attrs["class"])
    field_cls = ElementBase._subclasses[class_name]

    # load the instance from hdf
    return field_cls.from_hdf_dataset(dataset)


def element_from_file(path: str) -> ElementBase:
    """create element instance from a stored state

    Args:
        path (str): Path to the file being read
    """
    import h5py

    with h5py.File(path, "r") as fp:
        return element_from_hdf(fp)
