"""
Provides classes that track simulation results

.. autosummary::
   :nosignatures:

   ~FieldTracker
   ~DropletElementTracker

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from typing import Union

from droplets.droplet_tracks import DropletTrack, DropletTrackList
from droplets.emulsions import EmulsionTimeCourse
from pde.fields.base import FieldBase, GridBase
from pde.trackers.base import InfoDict, IntervalData, TrackerBase

from .simulation import State


class FieldTracker(TrackerBase):
    """ tracker for analyzing a discretized field in a simulations
    
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
        self.interval = tracker.interval
        self._logger = logging.getLogger(self.__class__.__name__)

    def initialize(  # type: ignore
        self, state: State, info: InfoDict = None,
    ) -> float:
        """ initialize the tracker with information about the simulation
        
        Args:
            state (:class:`~sim.state.State`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation        
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        if isinstance(state[self.element_name], FieldBase):
            field = state[self.element_name]
        elif hasattr(state[self.element_name], "_field") and isinstance(
            state[self.element_name]._field, FieldBase
        ):
            field = state[self.element_name]._field
        else:
            self._logger.warning(
                f"Element `{self.element_name}` does not seem "
                "to contain a scalar field"
            )
        return self.tracker.initialize(field, info)

    def handle(self, state: State, t: float) -> None:  # type: ignore
        """ handle data supplied to this tracker
        
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
                f"Element `{self.element_name}` does not seem "
                "to contain a scalar field"
            )
        self.tracker.handle(field, t)

    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation        
        """
        self.tracker.finalize(info)


class DropletElementTracker(TrackerBase):
    """ Tracker storing information about droplets in a simulation
    
    Attributes:
        emulsions (:class:`~droplets.analysis.emulsions.EmulsionTimeCourse`):
            An object describing the emulsion at the determined intervals
        droplet_tracks (:class:`~droplets.analysis.droplets.DropletTrackList`):
            An object describing the time course of individual droplets.
            
    The two attributes `emulsions` and `droplet_tracks` contain equivalent
    information, but their structure is different and either one might thus be
    used to analyze the simulation.
    """

    def __init__(
        self,
        element_name: str,
        interval: IntervalData = 1,
        store_emulsions: Union[bool, str] = True,
        store_droplet_tracks: Union[bool, str] = True,
        keep_vanished: bool = False,
        background_grid: GridBase = None,
    ):
        """
        Args:
            element_name (str):
                The name of the element containing the droplets
            interval
                Determines how often the tracker interrupts the simulation.
                Simple numbers are interpreted as durations measured in the
                simulation time variable. Alternatively, instances of
                :class:`~droplets.simulation.trackers.LogarithmicIntervals` and
                :class:`~droplets.simulation.trackers.RealtimeIntervals`
                might be given for more control.
            store_emulsions (bool or str):
                Determines whether to store data on emulsions in an instance of
                :class:`~droplets.analysis.emulsions.EmulsionTimeCourse`. No
                data is stored when this is `False`. Otherwise, the data is
                available in the :attr:`emulsions` attributed of the tracker
                instance. The data is additionally written to a file when a
                path is supplied as a string.
            store_droplet_tracks (bool or str):
                Determines whether to store data on droplets in an instance of
                :class:`~droplets.analysis.droplets.DropletTrackList`. No
                data is stored when this is `False`. Otherwise, the data is
                available in the :attr:`droplet_tracks` attributed of the
                tracker instance. The data is additionally written to a file
                when a path is supplied as a string.
            keep_vanished (bool):
                Flag determining whether vanished droplets (with zero radius)
                are still stored. The default is to filter these droplets.
            background_grid (:class:`pde.grids.base.GridBase`):
                The grid on which the droplets are defined. This is stored in
                the emulsion object to calculate distances and other geometric 
                quantities. 
        """
        super().__init__(interval=interval)
        self.element_name = element_name
        self.store_emulsions = store_emulsions
        self.store_droplet_tracks = store_droplet_tracks
        self.keep_vanished = keep_vanished
        self.background_grid = background_grid

    def initialize(  # type: ignore
        self, state: State, info: InfoDict = None,
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
            self.droplet_tracks = DropletTrackList()

        return super().initialize(state, info)  # type: ignore

    def handle(self, state: State, t: float) -> None:  # type: ignore
        """ handle data supplied to this tracker
        
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

        # handle droplet track list
        if self.store_droplet_tracks is not False:
            if len(self.droplet_tracks) == 0:
                # initialize droplet tracks
                for droplet in droplets:
                    track = DropletTrack(droplets=[droplet.copy()], times=[t])
                    self.droplet_tracks.append(track)
            else:
                # append to existing tracks
                for i, droplet in enumerate(droplets):
                    if self.keep_vanished or self.droplet_tracks[i].last.radius > 0:
                        self.droplet_tracks[i].append(droplet, time=t)

    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

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


__all__ = ["FieldTracker", "DropletElementTracker"]
