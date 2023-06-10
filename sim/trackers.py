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
from typing import Optional, Union

from droplets.droplet_tracks import DropletTrack, DropletTrackList
from droplets.emulsions import EmulsionTimeCourse
from modelrunner.state.trajectory import Trajectory as _Trajectory
from modelrunner.state.trajectory import TrajectoryWriter
from pde.fields.base import FieldBase, GridBase
from pde.tools.docstrings import fill_in_docstring
from pde.trackers.base import InfoDict, IntervalData, TrackerBase

from .simulation import State


class TrajectoryTracker(TrackerBase):
    """stores the state as a function of time during the simulation

    Stored data can be read using :class:`Trajectory`.
    """

    @fill_in_docstring
    def __init__(
        self,
        store,
        interval: IntervalData = 1,
        *,
        overwrite: bool = False,
        info: Optional[InfoDict] = None,
    ):
        """
        Args:
            store (MutableMapping or string):
                Store or path to directory in file system or name of zip file.
            interval
                {ARG_TRACKER_INTERVAL}
            overwrite (bool):
                If True, delete all pre-existing data in store.
            info (dict):
                Additional information that are written to the trajectory storage. To
                document simulation parameters, `simulation.info` can be used here.
        """
        super().__init__(interval=interval)
        self.store = store
        self.overwrite = overwrite
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
            info_write = copy.deepcopy(self.info)  # type: ignore
            info_write.update(info)

        self._writer = TrajectoryWriter(
            self.store, attrs=info_write, overwrite=self.overwrite
        )

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
        return self._state_attributes


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
        interval: IntervalData = 1,
        *,
        store_emulsions: Union[bool, str] = True,
        store_droplet_tracks: Union[bool, str] = True,
        keep_vanished: bool = False,
        background_grid: Optional[GridBase] = None,
    ):
        """
        Args:
            element_name (str):
                The name of the element containing the droplets
            interval
                {ARG_TRACKER_INTERVAL}
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
                droplets can disappear and re-appear in the simulation.
            background_grid (:class:`pde.grids.base.GridBase`):
                The grid on which the droplets are defined. This is stored in the
                emulsion object to calculate distances and other geometric quantities.
        """
        super().__init__(interval=interval)
        self.element_name = element_name
        self.store_emulsions = store_emulsions
        self.store_droplet_tracks = store_droplet_tracks
        self.keep_vanished = keep_vanished
        self.background_grid = background_grid

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
            self.emulsions.grid = self.background_grid

        if self.store_droplet_tracks is not False:
            tracks = [DropletTrack() for _ in range(len(state[self.element_name]))]
            self.droplet_tracks = DropletTrackList(tracks)

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
            for i, droplet in enumerate(droplets):
                track = self.droplet_tracks[i]
                is_active = track and track.last.radius > 0
                if is_active or droplet.radius > 0 or self.keep_vanished:
                    track.append(droplet, time=t)

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
        try:
            self.interrupt = tracker.interrupt
        except AttributeError:
            # fall-back to deprecated attribute (remove on 2023-03-15)
            self.interval = tracker.interval  # type: ignore
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
