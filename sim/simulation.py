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
from typing import (  # @UnusedImport
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numba as nb
import numpy as np

from pde.solvers.base import SolverBase
from pde.solvers.controller import Controller, TrackerCollectionDataType, TRangeType
from pde.solvers.explicit import ExplicitSolver
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
        actors: Optional[Sequence[Tuple[ElementNamesType, ActorBase]]] = None,
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
                result is available via the `timing` property of :class:`Simulation`,
                which contains the runtime of all actors in seconds.
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
        if actor.num_elements > 0 and len(elements) != actor.num_elements:
            raise ValueError(
                f"Actor {actor.__class__.__name__} expects {actor.num_elements} "
                f"elements, but {len(elements)} were given."
            )

        if check != "ignore":
            # run some checks before adding the actor

            def show_msg(msg: str, exception: TypeError):
                """helper function showing the message according to chosen method"""
                if check == "warn":
                    warnings.warn(msg)
                elif check == "log":
                    self._logger.warning(msg)
                elif check == "raise":
                    raise exception(msg)
                else:
                    raise ValueError(f"Unknown argument check='{check}'")

            if len(actor.element_classes) > 0:
                element_objects = [self.state.elements[name] for name in elements]
                if not actor.supports_elements(*element_objects, silent=True):
                    show_msg(f"Unsupported elements for `{actor.__class__.__name__}`")

            # check whether the same actor has already been added earlier
            for elements2, actor2 in self.actors:
                if elements2 == elements and actor2.__class__ is actor.__class__:
                    show_msg(
                        f"An actor of type `{actor.__class__.__name__}` has already "
                        f"been added for elements {elements}",
                        RuntimeError,
                    )

        self.actors.append((elements, actor))

    def get_graph(self, with_data: bool = True):
        """return a graph representation of the simulation

        Args:
            with_data (bool):
                Flag determining whether the element and actor objects are added to the
                vertices.

        Returns:
            :class:`networkx.DiGraph`: A graph where all elements and actors are
            represented as nodes.
        """
        from networkx import DiGraph

        graph = DiGraph()

        for name, element in self.state:
            if with_data:
                graph.add_node(f"element_{name}", obj=element, label=name)
            else:
                graph.add_node(f"element_{name}", label=name)

        for actor_id, (element_names, actor) in enumerate(self.actors, 1):
            actor_name = f"actor_{actor_id}"
            if with_data:
                graph.add_node(actor_name, obj=actor, label=actor.__class__.__name__)
            else:
                graph.add_node(actor_name, label=actor.__class__.__name__)
            for element_name in element_names:
                graph.add_edge(actor_name, f"element_{element_name}")

        return graph

    def plot_as_graph(self, layout: Union[str, Callable] = "auto", **kwargs) -> None:
        """represent the simulation in a graphical form

        Args:
            layout (str):
                Choose a method for determining the layout of the graph. Possible
                arguments include the names of all `nx.*_layout` functions.
            **kwargs:
                All arguments are passed to :func:`networkx.draw`
        """
        import networkx as nx

        graph = self.get_graph()

        # determine the layout of the graph
        if callable(layout):
            pos = layout(graph)
        elif layout == "auto":
            try:
                pos = nx.nx_pydot.pydot_layout(graph)
            except ImportError:
                _logger.warning("Suboptimal graph layout since `pydot` is unavailable")
                pos = nx.spring_layout(graph)
        else:
            pos = getattr(nx, layout + "_layout")(graph)

        # draw all nodes
        node_color = [
            "C1" if name.startswith("element") else "C0" for name in graph.nodes
        ]
        kwargs.setdefault("node_size", 1000)
        kwargs.setdefault("node_color", node_color)
        nx.draw(graph, pos, **kwargs)

        # label the nodes
        labels = {k: v["label"] for k, v in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels)

    def get_interacting_elements(self, with_data: bool = True):
        """return a graph representation the interacting elements of a simulation

        Args:
            with_data (bool):
                Flag determining whether the element and actor objects are added to the
                vertices and edges, respectively.

        Returns:
            :class:`networkx.DiGraph`: A graph where all elements are represented as
            nodes and their interactions are represented as edges.
        """
        from networkx import Graph

        graph = Graph()

        for name, element in self.state:
            if with_data:
                graph.add_node(name, element=element)
            else:
                graph.add_node(name)

        for elements, actor in self.actors:
            label = actor.__class__.__name__
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    if with_data:
                        graph.add_edge(
                            elements[i], elements[j], actor=actor, label=label
                        )
                    else:
                        graph.add_edge(elements[i], elements[j], label=label)

        return graph

    def plot_interacting_elements(
        self,
        layout: Union[str, Callable] = "auto",
        *,
        label_edges: bool = True,
        **kwargs,
    ) -> None:
        """plot all interacting elements as a graph

        Args:
            layout (str):
                Choose a method for determining the layout of the graph. Possible
                arguments include the names of all `nx.*_layout` functions or a callable
                function
            label_edges (bool):
                Flag determining whether the edges are labeled with the actors
            **kwargs:
                All arguments are passed to :func:`networkx.draw`
        """
        import networkx as nx

        graph = self.get_interacting_elements(with_data=False)

        # determine the layout of the graph
        if callable(layout):
            pos = layout(graph)
        elif layout == "auto":
            try:
                pos = nx.nx_pydot.pydot_layout(graph)
            except ImportError:
                _logger.warning("Suboptimal graph layout since `pydot` is unavailable")
                pos = nx.spring_layout(graph)
        else:
            pos = getattr(nx, layout + "_layout")(graph)

        # draw all nodes
        kwargs.setdefault("with_labels", True)
        kwargs.setdefault("node_color", "tab:orange")
        nx.draw(graph, pos, **kwargs)

        if label_edges:
            edge_labels = {(n1, n2): d["label"] for n1, n2, d in graph.edges(data=True)}
            nx.draw_networkx_edge_labels(
                graph, pos, edge_labels=edge_labels, label_pos=0.5
            )

    def estimate_dt(self, state: Optional[State] = None) -> float:
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

    def _make_evolve_state(
        self, actor_id: int, state: Optional[State] = None
    ) -> Callable[[Tuple[np.ndarray, ...], float, float], Union[float, None]]:
        """factory function creating a function to evolve a single actor

        Args:
            actor_id (int):
                The id of the actor that needs to be evolved
            state (:class:`~sim.state.State`):
                A state defining the degrees of freedom of the simulation.

        Returns:
            callable: a function that
        """
        # Programmer's note: We separated out this part of creating the inner evolvers
        # because of python's variable scoping. If the `evolve_state` functions were to
        # be defined in the inner loop in the the `make_evolver_numba` function the
        # variables used in `evolve_state` would always refer to the values at the last
        # loop (unless the whole function is compiled by numba). This leads to
        # unexpected behavior, so we now properly close the variables using this factory
        # function
        if state is None:
            state = self.state

        state_data_type = nb.typeof(state.data)

        elements, actor = self.actors[actor_id]
        actor_evolver = actor.make_evolver_numba(state[elements])
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
                actor_evolver(get_element_states(state_data), t, dt)
                with nb.objmode(runtime="f8"):
                    runtime = time.perf_counter() - time_start
                return runtime

        else:

            @jit(nb.none(state_data_type, nb.float64, nb.float64))
            def evolve_state(
                state_data: Tuple[np.ndarray, ...], t: float, dt: float
            ) -> None:
                """evolve the states affected by this actor"""
                states = get_element_states(state_data)
                actor_evolver(states, t, dt)

        # return the evolver for this actor, which now will be properly closed
        return evolve_state  # type: ignore

    def make_evolver_numba(self, state: Optional[State] = None) -> EvolverType:
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

        def chain(actor_id: int, inner: Optional[Callable] = None) -> Callable:
            """recursive factory function for running all actors"""
            # get the evolver function
            actor_evolver = self._make_evolve_state(actor_id, state=state)

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
                    timings[actor_id] += actor_evolver(state_data, t, dt)

            else:

                @jit
                def wrap(
                    state_data: Tuple[np.ndarray, ...], t: float, dt: float
                ) -> None:
                    if inner is not None:
                        inner(state_data, t, dt)
                    actor_evolver(state_data, t, dt)

            if actor_id < len(self.actors) - 1:
                # there are more items in the chain
                return chain(actor_id + 1, inner=wrap)
            else:
                # this is the outermost function
                return wrap  # type: ignore

        if self.profile:
            # add code recording the profiling timings

            self._logger.info("Construct the main evolver with timing information")
            self.timings = np.zeros(len(self.actors))  # initialize timing information
            get_timings_arr = make_array_constructor(self.timings)
            evolver_chain = chain(0)  # compile the recursive chain

            @jit
            def evolver(state_data: Tuple[np.ndarray, ...], t: float, dt: float):
                """wrapper to providing access to the timings array"""
                timings = get_timings_arr()
                evolver_chain(state_data, t, dt, timings)

            # prevent garbage collection of array
            evolver._timings = self.timings

        else:
            # construct the normal evolver using recursion
            self._logger.info("Construct the main evolver")
            evolver_chain = chain(0)  # compile the recursive chain
            evolver = jit(evolver_chain)

        return evolver  # type: ignore

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
        dt: Optional[float] = None,
        tracker: TrackerCollectionDataType = ["progress"],
        backend: str = "auto",
        ret_info: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[State, Tuple[State, Dict[str, Any]]]:
        r"""run the simulation to advance the state in time

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
            **kwargs:
                All additional arguments are forwarded to :class:`SimulationSolver`

        Returns:
            :class:`SimulationState`:
                The state of the simulation at the last time point. In the case
                `ret_info == True`, a tuple with the final state and a
                dictionary with additional information is returned.
        """
        cache_key = {"backend": backend, **kwargs}
        if (
            use_cache
            and "solver" in self._cache
            and self._cache["solver"]._cache_key == cache_key  # type: ignore
        ):
            # use the solver from the cache
            self._logger.info("Use cached solver")
            solver = self._cache["solver"]
        else:
            # create a new solver
            solver = SimulationSolver(
                self, backend=backend, use_cache=use_cache, **kwargs
            )
            solver._cache_key = cache_key  # type: ignore
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

    dt_min: float = 1e-10
    """float: minimal time step that the adaptive solver will use"""
    dt_max: float = 1e10
    """float: maximal time step that the adaptive solver will use"""

    def __init__(
        self,
        simulation: Simulation,
        *,
        backend: str = "auto",
        adaptive: bool = False,
        tolerance: float = 1e-4,
        use_cache: bool = False,
    ):
        """initialize the explicit solver for the actor-based simulation

        Args:
            simulation (:class:`Simulation`):
                The simulation that will be run
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
            adaptive (bool):
                When enabled, the time step is adjusted during the simulation using the
                error tolerance set with `tolerance`.
            tolerance (float):
                The error tolerance used in adaptive time stepping. This is used in
                adaptive time stepping to choose a time step which is small enough so
                the truncation error of a single step is below `tolerance`.
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
        self.adaptive = adaptive
        self.tolerance = tolerance
        self.use_cache = use_cache
        self._cache_stepper: Dict[str, Callable] = {}

    _make_dt_adjuster = ExplicitSolver._make_dt_adjuster

    def _make_fixed_stepper_numpy(self, dt: float) -> Callable:
        """return function evolving state from time `t_start` to `t_end`

        Args:
            dt (float): The time step

        Returns:
            callable: Function with signature (state: SimulationState,
            t_start: float, t_end: float), which advances `state` in time.
        """
        assert not self.adaptive

        if self.use_cache and "fixed_numpy" in self._cache_stepper:
            self._logger.info("Use cached fixed numpy stepper")
            return self._cache_stepper["fixed_numpy"]

        def stepper(state: State, t_start: float, t_end: float) -> float:
            """function that advances the state from t_start to t_end"""
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                self.simulation.evolve(state, t, dt)

            self.info["steps"] += steps

            return t + dt

        self._cache_stepper["fixed_numpy"] = stepper
        return stepper

    def _make_adaptive_stepper_numpy(self, dt: float) -> Callable:
        """return function evolving state using adaptive time steps

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        assert self.adaptive

        if self.use_cache and "adaptive_numpy" in self._cache_stepper:
            self._logger.info("Use cached adaptive numpy stepper")
            return self._cache_stepper["adaptive_numpy"]

        # obtain auxiliary functions
        adjust_dt = self._make_dt_adjuster()  # type: ignore
        tolerance = self.tolerance
        dt_min = self.dt_min

        dt_arr = np.array(dt)
        if self.backend == "numba":
            get_dt = make_array_constructor(dt_arr)
        elif self.backend == "numpy":
            get_dt = lambda: dt_arr
        else:
            raise NotImplementedError

        def stepper(state: State, t_start: float, t_end: float, dt_init=None) -> float:
            """create the adaptive stepper"""
            # get
            if dt_init is None:
                dt_opt = get_dt().item()
            else:
                dt_opt = dt_init

            t = t_start
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = np.clip(dt_opt, dt_min, t_end - t)

                state1 = state.copy()
                state2 = state.copy()

                # single step with current value for dt
                self.simulation.evolve(state1, t, dt_step)

                # double step with half the time step
                self.simulation.evolve(state2, t, 0.5 * dt_step)
                self.simulation.evolve(state2, t + 0.5 * dt_step, 0.5 * dt_step)

                # calculate maximal error
                error = 0.0
                for data1, data2 in zip(state1.data, state2.data):
                    error_item = np.abs(data1 - data2).max()
                    if np.isnan(error_item):
                        error = np.nan
                        break
                    else:
                        error = max(error_item, error)

                else:
                    # error is finite
                    error_rel = error / tolerance  # normalize error to given tolerance

                    # do the step if the error is sufficiently small
                    if error_rel <= 1:
                        steps += 1
                        t += dt_step
                        for i, el_data in enumerate(state2.data):
                            state.data[i][...] = el_data

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel, t)
                else:
                    break  # return to the controller

            return t

        stepper._dt_arr = dt_arr  # type: ignore
        self._cache_stepper["adaptive_numpy"] = stepper
        return stepper

    def _make_fixed_stepper_numba(self, state: State, dt: float) -> Callable:
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
        assert not self.adaptive

        if self.use_cache and "fixed_numba" in self._cache_stepper:
            self._logger.info("Use cached fixed numba stepper")
            return self._cache_stepper["fixed_numba"]

        simulation_evolver = self.simulation.make_evolver_numba(state)

        def stepper(state: State, t_start: float, t_end: float) -> float:
            """function that advances the state from t_start to t_end"""
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            for step in range(steps):
                t = t_start + step * dt  # advance time
                simulation_evolver(state.data, t, dt)

            self.info["steps"] += steps

            return t + dt

        self._cache_stepper["fixed_numba"] = stepper
        return stepper

    def make_stepper(self, state: State, dt: Optional[float] = None) -> Callable:
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
                dt = 1.0
                self._logger.warning(
                    f"Time step could not be determined automatically. Using dt={dt}"
                )

        # store information about the simulation
        self.info["dt"] = dt
        self.info["steps"] = 0

        if self.adaptive:
            # adaptive time step
            if self.backend == "auto" or self.backend == "numpy":
                return self._make_adaptive_stepper_numpy(dt)
            elif self.backend == "numba":
                raise NotImplementedError(f"numba backend with adaptive dt")
            else:
                raise ValueError(f"Unknown backend `{self.backend}`")

        else:
            # fixed time step
            if self.backend == "auto":
                try:
                    return self._make_fixed_stepper_numba(state, dt)
                except NotImplementedError:
                    self._logger.warning(
                        "Numba backend is not implemented for all "
                        "parts of the simulation."
                    )
                    return self._make_fixed_stepper_numpy(dt)

            elif self.backend == "numba":
                return self._make_fixed_stepper_numba(state, dt)
            elif self.backend == "numpy":
                return self._make_fixed_stepper_numpy(dt)
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
