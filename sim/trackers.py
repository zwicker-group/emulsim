"""
Provides classes that track the state of the simulation

.. autosummary::
   :nosignatures:

   ~TrajectoryTracker
   ~Trajectory
   ~DropletElementTracker
   ~FieldTracker

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy
import logging
from typing import Dict, Optional, Union

from droplets.droplet_tracks import DropletTrack, DropletTrackList
from droplets.emulsions import EmulsionTimeCourse
from modelrunner.storage import ModeType, StorageID
from modelrunner.storage import Trajectory as _Trajectory
from modelrunner.storage import TrajectoryWriter
from pde.fields.base import FieldBase
from pde.tools.docstrings import fill_in_docstring
from pde.trackers.base import InfoDict, InterruptData, TrackerBase

from .simulation import State


class TrajectoryTracker(TrackerBase):
    """stores the state as a function of time during the simulation

    Stored data can be read using :class:`Trajectory`.
    """

    @fill_in_docstring
    def __init__(
        self,
        storage: StorageID,
        interrupts: InterruptData = 1,
        *,
        mode: Optional[ModeType] = None,
        info: Optional[InfoDict] = None,
        interval=None,
    ):
        """
        Args:
            storage (MutableMapping or string):
                Store or path to directory in file system
            interrupts
                {ARG_TRACKER_INTERRUPTS}
            mode (str or :class:`~modelrunner.storage.access_modes.AccessMode`):
                The file mode with which the storage is accessed. Determines allowed
                operations. The meaning of the special (default) value `None` depends on
                whether the file given by `store` already exists. If yes, a RuntimeError
                is raised, otherwise the choice corresponds to `mode="full"` and thus
                creates a new trajectory. If the file exists, use `mode="truncate"` to
                overwrite file or `mode="append"` to insert new data into the file.
            info (dict):
                Additional information that are written to the trajectory storage. To
                document simulation parameters, `simulation.info` can be used here.
        """
        super().__init__(interrupts=interrupts, interval=interval)
        self.storage = storage
        self.mode = mode
        self.info = info

    def initialize(  # type: ignore
        self, state: State, info: Optional[InfoDict] = None
    ) -> float:
        """
        Args:
            state (:class:`~sim.state.State`):
                The initial state of the simulation
            info (dict):
                Extra information for the simulation
        """
        if not isinstance(state, State):
            self._logger.warning("state is not of type `State`")

        if self.info is None:
            info_write = info
        else:
            info_write = copy.deepcopy(self.info)
            info_write.update(info)  # type: ignore

        self._writer = TrajectoryWriter(self.storage, mode=self.mode, attrs=info_write)

        return super().initialize(state, info)  # type: ignore

    def handle(self, state: State, t: float) -> None:  # type: ignore
        """handle data supplied to this tracker

        Args:
            state (:class:`~sim.state.State`):
                The current state of the simulation
            t (float):
                The associated time
        """
        self._writer.append(state, time=t)

    def finalize(self, info: Optional[InfoDict] = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        self._writer.close()


# subclass to change the docstring
class Trajectory(_Trajectory):
    """Reads trajectories of states written with :class:`TrajectoryTracker`

    The class permits direct access to indivdual states using the square bracket
    notation. It is also possible to directly iterate over all states.

    Attributes:
        times (:class:`~numpy.ndarray`): Time points at which data is available
    """

    @property
    def info(self) -> InfoDict:
        """dict: information that was stored with the trajectory"""
        return self.attrs


class DropletElementTracker(TrackerBase):
    """stores information about droplets in a simulation

    Attributes:
        emulsions (:class:`~droplets.analysis.emulsions.EmulsionTimeCourse`):
            An object describing the emulsion at the determined intervals
        droplet_tracks (:class:`~droplets.analysis.droplets.DropletTrackList`):
            An object describing the time course of individual droplets.

    The two attributes `emulsions` and `droplet_tracks` contain equivalent
    information, but their structure is different and either one might thus be
    used to analyze the simulation.
    """

    @fill_in_docstring
    def __init__(
        self,
        element_name: str,
        interrupts: InterruptData = 1,
        *,
        store_emulsions: Union[bool, str] = True,
        store_droplet_tracks: Union[bool, str] = True,
        keep_vanished: bool = False,
        interval=None,
    ):
        """
        Args:
            element_name (str):
                The name of the element containing the droplets
            interrupts
                {ARG_TRACKER_INTERRUPTS}
            store_emulsions (bool or str):
                Determines whether to store data on emulsions in an instance of
                :class:`~droplets.analysis.emulsions.EmulsionTimeCourse`. No data is
                stored when this is `False`. Otherwise, the data is available in the
                :attr:`emulsions` attributed of the tracker instance. The data is
                additionally written to a file when a path is supplied as a string.
            store_droplet_tracks (bool or str):
                Determines whether to store data on droplets in an instance of
                :class:`~droplets.analysis.droplets.DropletTrackList`. No data is stored
                when this is `False`. Otherwise, the data is available in the
                :attr:`droplet_tracks` attributed of the tracker instance. The data is
                additionally written to a file when a path is supplied as a string.
            keep_vanished (bool):
                Flag determining whether vanished droplets (with zero radius) are still
                stored. The default is to filter these droplets. Enable this flag if
                droplets disappearing and re-appearing should be combined in a single
                track.
        """
        super().__init__(interrupts=interrupts, interval=interval)
        self.element_name = element_name
        self.store_emulsions = store_emulsions
        self.store_droplet_tracks = store_droplet_tracks
        self.keep_vanished = keep_vanished

    def initialize(  # type: ignore
        self, state: State, info: Optional[InfoDict] = None
    ) -> float:
        """
        Args:
            state (:class:`~sim.state.State`):
                The initial state of the simulation
            info (dict):
                Extra information for the simulation
        """
        if not isinstance(state, State):
            self._logger.warning("state is not of type `State`")

        # initialize the tracked data
        if self.store_emulsions is not False:
            self.emulsions = EmulsionTimeCourse()

        if self.store_droplet_tracks is not False:
            # tracks = [DropletTrack() for _ in range(len(state[self.element_name]))]
            if self.keep_vanished:
                self._active_tracks: Dict[int, int] = {
                    i: i for i in range(len(state[self.element_name]))
                }
            else:
                self._active_tracks = {}
            self.droplet_tracks = DropletTrackList()  # tracks)

        return super().initialize(state, info)  # type: ignore

    def handle(self, state: State, t: float) -> None:  # type: ignore
        """handle data supplied to this tracker

        Args:
            state (:class:`~sim.state.State`):
                The current state of the simulation
            t (float):
                The associated time
        """
        droplets = state[self.element_name].droplets

        # handle emulsion time course
        if self.store_emulsions is not False:
            # remove vanished droplets
            if self.keep_vanished:
                emulsion = droplets.copy()
            else:
                emulsion = droplets.copy(min_radius=0)
            # add emulsion without an additional copy
            self.emulsions.append(emulsion, time=t, copy=False)

        # append all droplets to existing tracks
        if self.store_droplet_tracks is not False:
            if self.keep_vanished:
                # extend all tracks, independent of droplet size
                for i, droplet in enumerate(droplets):
                    self.droplet_tracks[self._active_tracks[i]].append(droplet, time=t)
            else:
                # store information about all droplets with finite radius
                for i, droplet in enumerate(droplets):
                    if droplet.radius > 0:
                        # droplet of finite size
                        try:
                            track_id = self._active_tracks[i]
                        except KeyError:
                            # start new track
                            self._active_tracks[i] = len(self.droplet_tracks)
                            self.droplet_tracks.append(DropletTrack([droplet], [t]))
                        else:
                            # extend exisiting track
                            self.droplet_tracks[track_id].append(droplet, time=t)

                    elif i in self._active_tracks:
                        # droplet vanished =>  stop this track
                        del self._active_tracks[i]

    def finalize(self, info: Optional[InfoDict] = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        # write data to files, if requested
        if isinstance(self.store_emulsions, str):
            self.emulsions.to_file(self.store_emulsions)
        if isinstance(self.store_droplet_tracks, str):
            self.droplet_tracks.to_file(self.store_droplet_tracks)


class FieldTracker(TrackerBase):
    """wrapper to use `py-pde` trackers on fields

    This acts as a wrapper around any of the trackers from :mod:`pde.trackers`,
    e.g., `tracker = FieldTracker('background', PlotTracker())`.
    """

    def __init__(self, element_name: str, tracker: TrackerBase):
        """
        Args:
            element_name (str):
                The name of the element of the field
            tracker (TrackerBase):
                The tracker that will receive the field
        """
        self.element_name = element_name
        self.tracker = tracker
        self.interrupt = tracker.interrupt
        self._logger = logging.getLogger(self.__class__.__name__)

    def initialize(  # type: ignore
        self,
        state: State,
        info: Optional[InfoDict] = None,
    ) -> float:
        """initialize the tracker with information about the simulation

        Args:
            state (:class:`~sim.state.State`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        element = state[self.element_name]
        if isinstance(element, FieldBase):
            field = state[self.element_name]
        elif hasattr(element, "_field") and isinstance(element._field, FieldBase):
            field = state[self.element_name]._field
        else:
            raise RuntimeError(
                f"{element.__class__.__name__} `{self.element_name}` does not seem to "
                "contain a scalar field"
            )
        return self.tracker.initialize(field, info)

    def handle(self, state: State, t: float) -> None:  # type: ignore
        """handle data supplied to this tracker

        Args:
            state (:class:`~sim.state.State`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if isinstance(state[self.element_name], FieldBase):
            field = state[self.element_name]
        elif hasattr(state[self.element_name], "_field") and isinstance(
            state[self.element_name]._field, FieldBase
        ):
            field = state[self.element_name]._field
        else:
            self._logger.warning(
                f"Element `{self.element_name}` does not seem to contain a scalar field"
            )
        self.tracker.handle(field, t)

    def finalize(self, info: Optional[InfoDict] = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        self.tracker.finalize(info)


__all__ = ["TrajectoryTracker", "Trajectory", "DropletElementTracker", "FieldTracker"]
