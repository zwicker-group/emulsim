#!/usr/bin/env python3

import pathlib
import sys

# Root directory of the package
ROOT = pathlib.Path(__file__).absolute().parents[2]
# directory to which the documents are written
OUTPUT = ROOT / "docs" / "source" / "snippets"

sys.path.insert(0, str(ROOT))

from sim.actors.base import ActorBase
from sim.elements.base import _ElementBase


def main():
    """Parse all classes and write information in a special snippet text."""
    # create the output directory
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # obtain all classes
    actors = {
        cls: [] for cls in ActorBase._subclasses.values() if issubclass(cls, ActorBase)
    }

    elements = {
        cls: set()
        for cls in _ElementBase._subclasses.values()
        if issubclass(cls, _ElementBase)
    }

    # obtain elements compatible with an actor and vice versa
    for actor, connected_elements in actors.items():
        if actor.element_classes is Ellipsis:
            connected_elements.append(Ellipsis)
        else:
            for expected_element in actor.element_classes:
                supported_elements = []
                for el, connected_actors in elements.items():
                    if issubclass(el, expected_element):
                        supported_elements.append(el)
                        connected_actors.add(actor)
                connected_elements.append(supported_elements)

    # create the output file for the elements
    with (OUTPUT / "elements.rst").open("w") as fp:
        for element in sorted(elements, key=lambda e: e.__name__):
            if elements[element]:
                classname = element.__module__ + "." + element.__name__
                description = element.__doc__.split("\n", 1)[0][:-1]
                actors_str = "\n".join(
                    f"   - :class:`~{actor.__module__ + "." + actor.__name__}`"
                    for actor in sorted(elements[element], key=lambda a: a.__name__)
                )
                fp.write(
                    f"- :class:`~{classname}` ({description}):\n\n{actors_str}\n\n"
                )

    # create the output file for the actors
    with (OUTPUT / "actors.rst").open("w") as fp:
        for actor in sorted(actors, key=lambda a: a.__name__):
            connected_elements = actors[actor]
            if not connected_elements:
                continue
            classname = actor.__module__ + "." + actor.__name__
            description = actor.__doc__.split("\n", 1)[0][:-1]
            fp.write(f"- :class:`~{classname}` ({description}):\n\n")
            for i, elementlist in enumerate(connected_elements, 1):
                if elementlist is Ellipsis:
                    fp.write(f"  * Variable number of elements")
                else:
                    el_str = ", ".join(
                        f":class:`~{element.__module__ + "." + element.__name__}`"
                        for element in elementlist
                    )
                    fp.write(f"   - {el_str}\n")
            fp.write("\n")


if __name__ == "__main__":
    main()
