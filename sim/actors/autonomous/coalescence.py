"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import literal_unroll
from numba.extending import overload

from pde.tools.numba import jit

from ...elements import MulticomponentDropletsElement, SphericalDropletsElement
from ..base import ActorBase, ElementsType


def fill_with_zeros(recarr: np.recarray) -> None:
    """Fill a record array with zeros."""
    recarr.fill(0)


@overload(fill_with_zeros)
def ol_fill_with_zeros(recarr: np.recarray) -> Callable[[np.recarray], None]:
    """Create numba implementation to fill a record array with zeros."""
    keys = tuple(recarr.dtype.fields.keys())  # type: ignore

    def fill_with_zeros_impl(recarr: np.recarray) -> None:
        """Numba implementation to fill a record array with zeros."""
        for key in literal_unroll(keys):
            if isinstance(recarr[key], (int, float)):  # noqa: UP038 (no numba support)
                recarr[key] = 0
            else:
                recarr[key][:] = 0

    return fill_with_zeros_impl


class CoalescenceDropletActor(ActorBase):
    """Actor merging overlapping droplets."""

    element_classes = ((SphericalDropletsElement, MulticomponentDropletsElement),)

    def make_evolver_numba(
        self, elements: ElementsType
    ) -> Callable[[tuple[np.ndarray, ...], float, float], None]:
        """Return a function evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that contains the droplet information

        Returns:
            callable: A function with signature
                (field_data: :class:`~numpy.ndarray`, t: float, dt: float),
                evolving `field_data`
        """
        droplet_class = elements[0].droplet_class  # type: ignore
        merge_data = droplet_class._make_merge_data()

        @jit
        def evolver(state_data: tuple[np.ndarray], t: float, dt: float):
            """Evolve all points explicitly."""
            (data,) = state_data

            # sort all droplets by radius
            radii = np.array([droplet.radius for droplet in data])
            indices = np.argsort(radii)

            # run through droplets from smallest to largest
            for progress, i1 in enumerate(indices):
                if radii[i1] == 0:
                    continue  # skip vanished droplets

                # compare this droplet to all larger droplets
                for i2 in indices[progress + 1 :]:
                    dist = np.linalg.norm(data[i1].position - data[i2].position)
                    if dist < radii[i1] + radii[i2]:
                        # overlapping droplets -> remove smaller droplet
                        merge_data(data[i1], data[i2], out=data[i2])
                        fill_with_zeros(data[i1])
                        break  # droplet with index i1 has been removed -> continue

        return evolver  # type: ignore

    def evolve(self, elements: ElementsType, t: float, dt: float) -> None:
        """Evolve the field state from time `t` to `t + dt`

        Args:
            elements (tuple of :class:`~sim.elements.droplets.SphericalDropletsElement`):
                The field element that contains the droplet information
            t (float):
                The current time point
            dt (float):
                The time step
        """
        drop_el: MulticomponentDropletsElement = elements[0]  # type: ignore
        droplets = drop_el.droplets
        positions = droplets.data["position"]  # type: ignore
        radii = droplets.data["radius"]  # type: ignore

        # sort all droplets by radius
        indices = np.argsort(radii)

        # run through droplets from smallest to largest
        for progress, i1 in enumerate(indices):
            if radii[i1] == 0:
                continue  # skip vanished droplets

            # compare this droplet to all larger droplets
            for i2 in indices[progress + 1 :]:
                dist = np.linalg.norm(positions[i1] - positions[i2])
                if dist < radii[i1] + radii[i2]:
                    # overlapping droplets -> remove smaller droplet
                    drop1, drop2 = droplets[i1], droplets[i2]
                    # we explicitly convert to recarrays since some versions of numpy
                    # apparently return normal arrays here
                    drop2._merge_data(
                        drop1.data.view(type=np.recarray),
                        drop2.data.view(type=np.recarray),
                        out=drop2.data.view(type=np.recarray),
                    )
                    fill_with_zeros(drop1.data)
                    break  # droplet with index i1 has been removed -> continue
