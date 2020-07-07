"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from typing import Callable, Tuple

from pde.tools.numba import jit
from pde.tools import spherical
from ...elements import SphericalDropletsElement

from ..base import ActorBase, ElementsType


class CoalescenceDropletActor(ActorBase):
    """ represents actor that moves droplets according to Brownian motion """

    element_classes = (SphericalDropletsElement,)

    def make_evolver_numba(self, elements: ElementsType) -> Callable:
        """ return a function evolve the field state from time `t` to `t + dt`
        
        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that is effected by the Brownian motion

        Returns:
            callable: A function with signature
                (field_data: :class:`numpy.ndarray`, t: float, dt: float),
                evolving `field_data`
        """

        dim = elements[0].dim
        radius = spherical.make_radius_from_volume_compiled(dim)
        volume = spherical.make_volume_from_radius_compiled(dim)

        @jit
        def evolver(state_data: Tuple[np.ndarray], t: float, dt: float):
            """ evolve all points explicitly """
            (data,) = state_data

            # sort all droplets by radius
            radii = np.array([droplet.radius for droplet in data])
            indices = np.argsort(radii)

            # run through droplets from smallest to largest
            for i, drop1 in enumerate(indices):
                if radii[drop1] == 0:
                    continue  # skip vanished droplets

                # compare this droplet to all larger droplets
                for drop2 in indices[i + 1 :]:
                    dist = np.linalg.norm(data[drop1].position - data[drop2].position)
                    if dist < radii[drop1] + radii[drop2]:  # overlapping droplets
                        vol = volume(data[drop1].radius) + volume(data[drop2].radius)
                        data[drop1].radius = 0  # remove first droplet
                        data[drop2].radius = radius(vol)
                        # TODO: adjust position
                        break

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """ evolve the field state from time `t` to `t + dt`
        
        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that is effected by the Brownian motion
            t (float):
                The current time point
            dt (float):
                The time step
        """
        droplets = elements[0].droplets  # type: ignore
        positions = droplets.data["position"]
        radii = droplets.data["radius"]

        # sort all droplets by radius
        indices = np.argsort(radii)

        # run through droplets from smallest to largest
        for i, drop1 in enumerate(indices):
            if radii[drop1] == 0:
                continue  # skip vanished droplets

            # compare this droplet to all larger droplets
            for drop2 in indices[i + 1 :]:
                dist = np.linalg.norm(positions[drop1] - positions[drop2])
                if dist < radii[drop1] + radii[drop2]:  # overlapping droplets
                    vol = droplets[drop1].volume + droplets[drop2].volume
                    droplets[drop1].radius = 0  # remove first droplet
                    droplets[drop2].volume = vol
                    # TODO: adjust position
                    break
