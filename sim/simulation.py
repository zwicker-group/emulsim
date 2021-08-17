"""
Provides a class representing the full simulation

.. autosummary::
   :nosignatures:

   ~Simulation
   ~SimulationSolver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union  # @UnusedImport

import numba as nb
import numpy as np

from pde.solvers.base import SolverBase
from pde.solvers.controller import Controller, TrackerCollectionDataType, TRangeType
from pde.tools.numba import jit, make_array_constructor

from .actors.base import ActorBase, EvolverType
from .state import State

_logger = logging.getLogger(__name__)

ElementNamesType = Union[str, Tuple[str]]


class Simulation:
    """Class defining the simulation state"""

    def __init__(
        self,
        state: State,
        actors: Sequence[Tuple[ElementNamesType, ActorBase]] = None,
        *,
        check: str = "log",
        profile: bool = False,
    ):
        """
        Args:
            state (:class:`~sim.state.State`):
                The initial simulation state defining the elements in the simulation.
            actors (sequence, optional):
                The actors in the simulation. This should be an iterable returning an
                `(element_names, actor)` pair for each item, where `element_names` is a
                sequence of all elements this actor affects. All actors are added to the
                simulation by calling :meth:`~Simulation.add_actor`.
            check (str):
                A flag determining what to do when the chosen elements are not the ones
                expected by the actor class. Possible options are: `ignore` (skip
                checks), `warn` (using :mod:`warnings` module), `log` (warn using
                :mod:`logging` module), or `raise` (raise a :class:`RuntimeError`).
            profile (bool):
                Flag indicating whether the simulation should be profiled. If True, the
                accumulated duration of each actor is recorded during a simulation. The
                result is available via the `timing` property of :class:`Simulation`.
        """
        self.state = state
        self._logger = logging.getLogger(self.__class__.__name__)
        self.actors: List[Tuple[ElementNamesType, ActorBase]] = []
        if actors is not None:
            for element_names, actor in actors:
                self.add_actor(element_names, actor, check=check)
        self.profile = profile
        self._cache: Dict[str, SimulationSolver] = {}

    def __repr__(self):
        """return instance as string"""
        actors_str = ", ".join(repr(actor) for actor in self.actors)
        return f"{self.__class__.__name__}({self.state!r}, actors=[{actors_str}])"

    def __str__(self):
        """return instance as string"""
        actors_str = ", ".join(str(actor) for actor in self.actors)
        return f"{self.__class__.__name__}({self.state!s}, actors=[{actors_str}])"

    @property
    def info(self) -> Dict[str, Any]:
        """dict: information about the state"""
        actor_infos = []
        for element_names, actor in self.actors:
            info = actor.info
            info["element_names"] = element_names
            actor_infos.append(info)
        return {"state": self.state.attributes, "actors": actor_infos}

    def add_actor(
        self, elements: Union[str, Tuple[str]], actor: ActorBase, *, check: str = "log"
    ):
        """adds a new actor to the simulation

        Args:
            elements (str or tuple of str):
                The elements this actor acts upon. This needs to have the exact number
                of elements the actor expects. In the special case of autonomous actors,
                a single string can be given instead of a tuple with a single entry.
            actor (:class:`~sim.actors.base.ActorBase`):
                The instance describing the actor.
            check (str):
                A flag determining what to do when the chosen elements are not the ones
                expected by the actor class. Possible options are: `ignore` (skip
                checks), `warn` (using :mod:`warnings` module), `log` (warn using
                :mod:`logging` module), or `raise` (raise a :class:`RuntimeError`).
        """
        if isinstance(elements, str):
            elements = (elements,)

        # check whether the chosen elements are actually in the state
        for element_name in elements:
            if element_name not in self.state.elements:
                raise ValueError(f'No element "{element_name}" in state')

        # check whether the number of elements agrees with what the actor expects
        if len(elements) != actor.num_elements:
            raise ValueError(
                f"Actor {actor.__class__.__name__} expects {actor.num_elements} "
                f"elements, but {len(elements)} were given."
            )

        if check != "ignore":
            # run some checks before adding the actor

            def show_msg(msg: str):
                """helper function showing the message according to chosen method"""
                if check == "warn":
                    warnings.warn(msg)
                elif check == "log":
                    self._logger.warning(msg)
                elif check == "raise":
                    raise RuntimeError(msg)
                else:
                    raise ValueError(f"Unknown argument check='{check}'")

            # check whether all elements have the expected type
            for element_name, element_class in zip(elements, actor.element_classes):
                element = self.state.elements[element_name]
                if hasattr(element_class, "__iter__"):
                    # actor supports multiple classes for this element
                    if not any(isinstance(element, cls) for cls in element_class):  # type: ignore
                        show_msg(
                            f"Element '{element_name}' is a "  # type: ignore
                            f"`{element.__class__.__name__}`, but actor type "
                            f"`{actor.__class__.__name__}` expects any of "
                            f"`{', '.join(cls.__name__ for cls in element_class)}`"
                        )

                else:
                    # actor supports a single class for this element
                    if not isinstance(element, element_class):
                        show_msg(
                            f"Element '{element_name}' is a "  # type: ignore
                            f"`{element.__class__.__name__}`, but actor type "
                            f"`{actor.__class__.__name__}` expects "
                            f"`{element_class.__name__}`"
                        )

            # check whether the same actor has already been added earlier
            for elements2, actor2 in self.actors:
                if elements2 == elements and actor2.__class__ is actor.__class__:
                    show_msg(
                        f"An actor of type `{actor.__class__.__name__}` has already "
                        f"been added for elements {elements}"
                    )

        self.actors.append((elements, actor))

    def get_graph(self):
        """return a graph representation of the simulation

        Returns:
            :class:`networkx.DiGraph`: A graph where all elements and actors are
            represented as nodes.
        """
        from networkx import DiGraph

        graph = DiGraph()

        for name, element in self.state:
            graph.add_node(f"element_{name}", obj=element, label=name)

        for actor_id, (element_names, actor) in enumerate(self.actors, 1):
            actor_name = f"actor_{actor_id}"
            graph.add_node(actor_name, obj=actor, label=actor.__class__.__name__)
            for element_name in element_names:
                graph.add_edge(actor_name, f"element_{element_name}")

        return graph

    def plot_as_graph(self, **kwargs) -> None:
        """represent the simulation in a graphical form

        Args:
            **kwargs:
                All arguments are passed to :func:`networkx.draw`
        """
        import networkx as nx

        graph = self.get_graph()

        # determine the layout of the graph
        try:
            pos = nx.nx_pydot.pydot_layout(graph)
        except ImportError:
            _logger.warning("Suboptimal graph layout since `pydot` is not available")
            pos = nx.spring_layout(graph)

        # draw all nodes
        node_color = [
            "tab:orange" if name.startswith("element") else "tab:blue"
            for name in graph.nodes
        ]
        kwargs.setdefault("node_size", 1000)
        kwargs.setdefault("node_color", node_color)
        nx.draw(graph, pos, **kwargs)

        # label the nodes
        labels = {k: v["label"] for k, v in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels)

    def get_interacting_elements(self):
        """return a graph representation the interacting elements of a simulation

        Returns:
            :class:`networkx.DiGraph`: A graph where all elements are represented as nodes
            and their interactions are represented as edges.
        """
        from networkx import Graph

        graph = Graph()

        for name, element in self.state:
            graph.add_node(name, element=element)

        for names, actor in self.actors:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    graph.add_edge(names[i], names[j], actor=actor)

        return graph

    def plot_interacting_elements(self, **kwargs) -> None:
        """plot all interacting elements as a graph"""
        import networkx as nx

        graph = self.get_interacting_elements()

        # determine the layout of the graph
        try:
            pos = nx.nx_pydot.pydot_layout(graph)
        except ImportError:
            pos = nx.spring_layout(graph)

        # draw all nodes
        kwargs.setdefault("with_labels", True)
        kwargs.setdefault("node_color", "tab:orange")
        nx.draw(graph, pos, **kwargs)

    def estimate_dt(self, state: State = None) -> float:
        """get the optimal time step for the simulation

        Args:
            state (:class:`~sim.state.State`):
                A state, which may influence the calculation of the time step

        Returns:
            float: the time step
        """
        if state is None:
            state = self.state

        dts: List[float] = [np.inf]
        for elements, actor in self.actors:
            try:
                dt = actor.estimate_dt(state[elements])
            except NotImplementedError:
                self._logger.info(f'Unknown time step for actor "{actor}"')
            else:
                dts.append(dt)

        return min(dts)

    def make_evolver_numba(self, state: State = None) -> EvolverType:
        """return a function evolving the state from time `t` to `t + dt`

        Args:
            state (:class:`~sim.state.State`):
                A state defining the degrees of freedom of the simulation.

        Returns:
            callable: A function with signature (state_data, t: float,
            dt: float), which evolves the state in time
        """
        if state is None:
            state = self.state

        state_data_type = nb.typeof(state.data)

        evolvers: List[Callable] = []
        for elements, actor in self.actors:

            # create the evolver for this actor
            evolver = actor.make_evolver_numba(state[elements])
            element_indices = tuple(state.get_index(name) for name in elements)
            get_element_states = make_get_element_states(element_indices)

            if self.profile:
                # add profiler information to the actor evolve function

                @jit(nb.float64(state_data_type, nb.float64, nb.float64))
                def evolve_state(
                    state_data: Tuple[np.ndarray, ...], t: float, dt: float
                ) -> float:
                    """evolve the states affected by this actor and record runtime"""
                    with nb.objmode(time_start="f8"):
                        time_start = time.perf_counter()
                    evolver(get_element_states(state_data), t, dt)
                    with nb.objmode(runtime="f8"):
                        runtime = time.perf_counter() - time_start
                    return runtime

            else:

                @jit(nb.none(state_data_type, nb.float64, nb.float64))
                def evolve_state(
                    state_data: Tuple[np.ndarray, ...], t: float, dt: float
                ):
                    """evolve the states affected by this actor"""
                    states = get_element_states(state_data)
                    evolver(states, t, dt)

            # store data for this actor
            evolvers.append(evolve_state)

        def chain(actor_id: int, inner: Callable = None) -> Callable:
            """recursive helper function for running all actors"""
            evolver = evolvers[actor_id]  # consider this particular evolver

            if self.profile:

                @jit
                def wrap(
                    state_data: Tuple[np.ndarray, ...],
                    t: float,
                    dt: float,
                    timings: np.ndarray,
                ) -> None:
                    if inner is not None:
                        inner(state_data, t, dt, timings)
                    timings[actor_id] += evolver(state_data, t, dt)

            else:

                @jit
                def wrap(
                    state_data: Tuple[np.ndarray, ...], t: float, dt: float
                ) -> None:
                    if inner is not None:
                        inner(state_data, t, dt)
                    evolver(state_data, t, dt)

            if actor_id < len(evolvers) - 1:
                # there are more items in the chain
                return chain(actor_id + 1, inner=wrap)
            else:
                # this is the outermost function
                return wrap  # type: ignore

        # compile the recursive chain
        evolver_chain = chain(0)

        # add code recording the profiling timings
        if self.profile:

            self._logger.info("Construct the main evolver with timing information")
            self.timings = np.zeros(len(self.actors))  # initialize timing information
            get_timings_arr = make_array_constructor(self.timings)

            @jit
            def evolver(state_data: Tuple[np.ndarray, ...], t: float, dt: float):
                """wrapper to providing access to the timings array"""
                timings = get_timings_arr()
                evolver_chain(state_data, t, dt, timings)

            # prevent garbage collection of array
            evolver._timings = self.timings  # type: ignore

        else:
            self._logger.info("Construct the main evolver")
            evolver = jit(evolver_chain)

        return evolver

    def evolve(self, state: State, t: float, dt: float) -> None:
        """evolve the state from time `t` to `t + dt`

        Args:
            state (:class:`~sim.state.State`):
                The state of the simulation
            t (float):
                The current time point
            dt (float):
                The time step
        """
        if self.profile:
            # record timing information
            if not hasattr(self, "timings"):
                self.timings = np.zeros(len(self.actors))

            for actor_id, (elements, actor) in enumerate(self.actors):
                time_start = time.perf_counter()
                actor.evolve(state[elements], t, dt)
                self.timings[actor_id] += time.perf_counter() - time_start

        else:
            # just evolve all actors
            for elements, actor in self.actors:
                actor.evolve(state[elements], t, dt)

    def run(
        self,
        t_range: TRangeType,
        dt: float = None,
        tracker: TrackerCollectionDataType = ["progress"],
        backend: str = "auto",
        ret_info: bool = False,
        use_cache: bool = False,
    ) -> Union[State, Tuple[State, Dict[str, Any]]]:
        """run the simulation to advance the state in time

        Args:
            t_range (float or tuple of floats):
                Sets the time range for which the simulation is run. If only a
                single value `t_end` is given, the time range is assumed to be
                `[0, t_end]`.
            dt (float):
                Time step of the explicit stepping. If `None`, the time step
                will be chosen automatically using the method
                :meth:`~Simulation.estimate_dt`.
            tracker:
                Defines trackers that process the state of the simulation at
                fixed time intervals. Multiple trackers can be specified as a
                list. The default value simply displays a progress bar. To
                disable trackers, set the value to `None`.
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
            ret_info (bool):
                Flag determining whether diagnostic information about the solver
                process should be returned.
            use_cache (bool):
                Indicates whether a stepper from the cache can also be used. This is
                disabled by default since there is no check whether the simulation
                parameters changed. However, using the cache can accelerate a second run
                of the simulation when the stepper are identical.

        Returns:
            :class:`SimulationState`:
                The state of the simulation at the last time point. In the case
                `ret_info == True`, a tuple with the final state and a
                dictionary with additional information is returned.
        """
        if (
            use_cache
            and "solver" in self._cache
            and self._cache["solver"].backend == backend
        ):
            # use the solver from the cache
            self._logger.info("Use cached solver")
            solver = self._cache["solver"]
        else:
            # create a new solver if it was not loaded from cache
            solver = SimulationSolver(self, backend=backend, use_cache=use_cache)
            self._cache["solver"] = solver

        # create a controller that handles trackers
        controller = Controller(solver, t_range=t_range, tracker=tracker)

        # run the actual simulation
        final_state: State = controller.run(self.state, dt)  # type: ignore

        if ret_info:
            info = controller.info.copy()
            info.pop("solver_class")  # remove redundant information
            info["solver"] = solver.info.copy()
            return final_state, info
        else:
            return final_state


class SimulationSolver(SolverBase):
    """Solver for actor-based simulation"""

    def __init__(
        self, simulation: Simulation, backend: str = "auto", use_cache: bool = False
    ):
        """initialize the explicit solver for the actor-based simulation

        Args:
            simulation (:class:`Simulation`):
                The simulation that will be run
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
            use_cache (bool):
                Indicates whether a stepper from the cache can also be used. This is
                disabled by default since there is no check whether the simulation
                parameters changed. However, using the cache can accelerate a second run
                of the simulation when the stepper are identical.
        """
        self.info: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self.simulation = simulation
        self.backend = backend
        self.use_cache = use_cache
        self._cache_stepper: Dict[str, Callable] = {}

    def _make_stepper_numpy(self, dt: float) -> Callable:
        """return function evolving state from time `t_start` to `t_end`

        Args:
            dt (float): The time step

        Returns:
            callable: Function with signature (state: SimulationState,
            t_start: float, t_end: float), which advances `state` in time.
        """
        if self.use_cache and "numpy" in self._cache_stepper:
            self._logger.info("Use cached numpy stepper")
            return self._cache_stepper["numpy"]

        def stepper(state: State, t_start: float, t_end: float) -> float:
            """function that advances the state from t_start to t_end"""
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                self.simulation.evolve(state, t, dt)

            self.info["steps"] += steps

            return t + dt

        self._cache_stepper["numpy"] = stepper
        return stepper

    def _make_stepper_numba(self, state: State, dt: float) -> Callable:
        """return function evolving state from time `t_start` to `t_end`

        This function uses compiled functions for the actors.

        Args:
            state (:class:`~sim.state.State`):
                A state determining the degrees of freedom. If `None`, the state
                given by `self.simulation` will be used.
            dt (float):
                The time step

        Returns:
            callable: Function with signature (state: SimulationState,
            t_start: float, t_end: float), which advances `state` in time.
        """
        if self.use_cache and "numba" in self._cache_stepper:
            self._logger.info("Use cached numba stepper")
            return self._cache_stepper["numba"]

        simulation_evolver = self.simulation.make_evolver_numba(state)

        def stepper(state: State, t_start: float, t_end: float) -> float:
            """function that advances the state from t_start to t_end"""
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                simulation_evolver(state.data, t, dt)

            self.info["steps"] += steps

            return t + dt

        self._cache_stepper["numba"] = stepper
        return stepper

    def make_stepper(self, state: State, dt: float = None) -> Callable:
        r"""return a stepper function using an explicit scheme

        Note that if the `numba` backend is chosen, the state supplied to this
        function must be the identical state that is also used in the stepper.

        Args:
            state (:class:`~sim.state.State`):
                An example of the simulation state, which defines the degrees of
                freedom of the simulation and supplies other information.
            dt (float):
                Time step of the explicit stepping. If `None`, the time step
                will be chosen automatically using the method
                :func:`~Simulation.estimate_dt`.
            \**kwargs: These are currently ignored

        Returns:
            Function that can be called to advance the `state` from time
            `t_start` to time `t_end`. The function call signature is
            `(state: AgentSimulation, t_start: float, t_end: float)`
        """
        if dt is None:
            dt = self.simulation.estimate_dt(state)
            if np.isinf(dt):
                # this can happen if there are no restrictions on the time step
                dt = 1e3

        # store information about the simulation
        self.info["dt"] = dt
        self.info["steps"] = 0

        if self.backend == "auto":
            try:
                return self._make_stepper_numba(state, dt)
            except NotImplementedError:
                self._logger.warning(
                    "Numba backend is not implemented for all "
                    "parts of the simulation."
                )
                return self._make_stepper_numpy(dt)

        elif self.backend == "numba":
            return self._make_stepper_numba(state, dt)
        elif self.backend == "numpy":
            return self._make_stepper_numpy(dt)
        else:
            raise ValueError(f"Unknown backend `{self.backend}`")


def make_get_element_states(
    element_indices: Tuple[int, ...]
) -> Callable[[Tuple[np.ndarray, ...]], Tuple[np.ndarray, ...]]:
    """creates helper function that extracts the states of the given elements

    Args:
        element_indices (tuple): Indices of the elements to be extracted
    """
    num_elements = len(element_indices)
    if num_elements == 1:
        i = element_indices[0]

        @jit
        def get_element_states(state_data: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray]:
            return (state_data[i],)

    elif num_elements == 2:
        i, j = element_indices

        @jit
        def get_element_states(
            state_data: Tuple[np.ndarray, ...]
        ) -> Tuple[np.ndarray, np.ndarray]:
            return (state_data[i], state_data[j])

    elif num_elements == 3:
        i, j, k = element_indices

        @jit
        def get_element_states(
            state_data: Tuple[np.ndarray, ...]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (state_data[i], state_data[j], state_data[k])

    elif num_elements == 4:
        i, j, k, l = element_indices

        @jit
        def get_element_states(
            state_data: Tuple[np.ndarray, ...]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sd = state_data
            return (sd[i], sd[j], sd[k], sd[l])

    else:
        raise NotImplementedError(f"{num_elements} elements in actor")

    return get_element_states  # type: ignore


__all__ = ["Simulation", "SimulationSolver"]
