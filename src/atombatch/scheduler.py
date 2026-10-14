"""Run many ASE tasks and batch their energy and force requests per model."""

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ase.calculators.calculator import Calculator, all_changes

from .backends import as_backend

_REQUIRED = ("energy", "forces")


@dataclass
class TaskResult:
    """Outcome of one task: its return value, or the exception it raised."""

    name: str
    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self):
        return self.error is None


@dataclass
class _Request:
    model: str
    atoms: Any
    properties: tuple
    result: dict | None = None
    error: BaseException | None = None


@dataclass
class _Call:
    """Requests one task thread waits on together."""

    requests: list
    event: threading.Event = field(default_factory=threading.Event)


class BatchCalculator(Calculator):
    """ASE calculator whose evaluations are batched by a :class:`Scheduler`.

    Each instance belongs to one task and one model. ``calculate`` submits a
    request and blocks the task's thread until the scheduler has evaluated
    it together with the requests of the other tasks.

    Calculators created together with :meth:`TaskContext.linked_calculators`
    form a group: a request from one member also evaluates every other member
    whose structure changed, so a task that queries several structures in
    turn, such as an NEB band, submits them as one request.
    """

    implemented_properties = [  # noqa: RUF012 (ASE calculator convention)
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    ]

    def __init__(self, scheduler, model, properties=_REQUIRED):
        super().__init__()
        self._scheduler = scheduler
        self._model = model
        self._properties = tuple(dict.fromkeys((*_REQUIRED, *properties)))
        self._group = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = tuple(dict.fromkeys((*self._properties, *properties)))
        members = [(self, atoms)]
        for calculator, other in self._group or ():
            if calculator is self or other.calc is not calculator:
                continue
            stale = calculator.check_state(other) or any(
                prop not in calculator.results for prop in requested
            )
            if stale:
                members.append((calculator, other))
        results = self._scheduler._evaluate(
            [(calculator._model, other, requested) for calculator, other in members]
        )
        for (calculator, other), result in zip(members, results):
            if calculator is not self:
                calculator.atoms = other.copy()
            calculator.results = result


class TaskContext:
    """Handle given to a task function to create batched calculators."""

    def __init__(self, scheduler, model):
        self._scheduler = scheduler
        self.model = model

    def calculator(self, model=None, properties=_REQUIRED):
        """Return a new batched calculator for ``model`` (default: the task's).

        Every request from the calculator includes ``properties``, so a task
        that reads several properties at each state (for example forces and
        stress under a cell filter) is evaluated once per state.
        """
        model = self.model if model is None else model
        if model not in self._scheduler._backends:
            raise KeyError(f"Unknown model {model!r}.")
        return BatchCalculator(self._scheduler, model, properties)

    def linked_calculators(self, structures, model=None, properties=_REQUIRED):
        """Attach one grouped calculator to each structure and return them."""
        calculators = [self.calculator(model, properties) for _ in structures]
        group = list(zip(calculators, structures))
        for calculator, atoms in group:
            calculator._group = group
            atoms.calc = calculator
        return calculators


@dataclass
class _Task:
    name: str
    function: Callable
    model: str
    thread: threading.Thread | None = None
    result: TaskResult | None = None


class Scheduler:
    """Run ASE tasks in worker threads and batch their model evaluations.

    Each task runs ordinary ASE code with :class:`BatchCalculator` instances.
    When every running task is waiting for a result, the scheduler evaluates
    all pending requests in batched calls, one or more per model, and resumes
    the tasks. At most ``max_active`` tasks run at once; finished tasks are
    replaced from the queue so that batches stay full.
    """

    def __init__(self, max_active=None):
        self.max_active = max_active
        self._backends = {}
        self._queue = deque()
        self._tasks = []
        self._active = set()
        self._pending = []
        self._waiting = 0
        self._condition = threading.Condition()
        self.batch_log = []

    def add_model(self, name, model, max_atoms=None):
        """Register a backend, a Symmetrix calculator, or any ASE calculator."""
        self._backends[name] = as_backend(model, max_atoms=max_atoms)

    def submit(self, function, model, name=None):
        """Queue ``function(context)``; its return value becomes the result."""
        if model not in self._backends:
            raise KeyError(f"Unknown model {model!r}; register it with add_model.")
        task = _Task(name or f"task-{len(self._tasks)}", function, model)
        self._tasks.append(task)
        self._queue.append(task)
        return task.name

    def run(self):
        """Run all queued tasks and return their results in submission order."""
        tasks = list(self._queue)
        self._admit()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: len(self._active) == self._waiting)
                if not self._active and not self._queue:
                    break
                calls, self._pending = self._pending, []
            if calls:
                self._serve(calls)
            self._admit()
        return [task.result for task in tasks]

    def _admit(self):
        with self._condition:
            while self._queue and (
                self.max_active is None or len(self._active) < self.max_active
            ):
                task = self._queue.popleft()
                self._active.add(task.name)
                task.thread = threading.Thread(
                    target=self._run_task, args=(task,), name=task.name, daemon=True
                )
                task.thread.start()

    def _run_task(self, task):
        result = TaskResult(task.name)
        task.result = result
        try:
            result.value = task.function(TaskContext(self, task.model))
        except Exception as error:  # noqa: BLE001 (reported in the task result)
            result.error = error
        finally:
            with self._condition:
                self._active.discard(task.name)
                self._condition.notify_all()

    def _evaluate(self, items):
        call = _Call(
            [
                _Request(model, atoms.copy(), properties)
                for model, atoms, properties in items
            ]
        )
        with self._condition:
            self._pending.append(call)
            self._waiting += 1
            self._condition.notify_all()
        call.event.wait()
        for request in call.requests:
            if request.error is not None:
                raise request.error
        return [request.result for request in call.requests]

    def _serve(self, calls):
        requests = [request for call in calls for request in call.requests]
        groups = {}
        for request in requests:
            stress = "stress" in request.properties
            groups.setdefault((request.model, stress), []).append(request)
        for (model, stress), group in groups.items():
            backend = self._backends[model]
            properties = (*_REQUIRED, "stress") if stress else _REQUIRED
            for chunk in _chunks(group, backend.max_atoms):
                self.batch_log.append(
                    (model, len(chunk), sum(len(r.atoms) for r in chunk))
                )
                try:
                    results = backend.evaluate([r.atoms for r in chunk], properties)
                    if len(results) != len(chunk):
                        raise RuntimeError(
                            f"Backend {model!r} returned {len(results)} results "
                            f"for {len(chunk)} structures."
                        )
                except Exception as error:  # noqa: BLE001 (raised in each task)
                    for request in chunk:
                        request.error = error
                    continue
                for request, result in zip(chunk, results):
                    result = dict(result)
                    result.setdefault("free_energy", result["energy"])
                    request.result = result
        with self._condition:
            self._waiting -= len(calls)
        for call in calls:
            call.event.set()


def _chunks(requests, max_atoms):
    if max_atoms is None:
        yield requests
        return
    chunk, atoms = [], 0
    for request in requests:
        size = len(request.atoms)
        if chunk and atoms + size > max_atoms:
            yield chunk
            chunk, atoms = [], 0
        chunk.append(request)
        atoms += size
    if chunk:
        yield chunk
