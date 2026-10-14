"""Unit tests of scheduler, calculator, and backend internals.

These tests isolate one component each and use small fakes instead of a
model, so they run in milliseconds and need neither torch nor an MLIP.
"""

import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from atombatch import (
    Backend,
    FunctionBackend,
    Scheduler,
    SequentialBackend,
    SymmetrixBackend,
    TaskResult,
    TorchSimBackend,
    single_point,
    swap_monte_carlo,
)
from atombatch.backends import as_backend
from atombatch.scheduler import _chunks, _Request


def _atoms(count):
    return Atoms(f"Cu{count}", positions=np.arange(3 * count).reshape(count, 3))


class Recording(Backend):
    """Backend that records each call and returns per-structure markers."""

    def __init__(self, max_atoms=None):
        self.max_atoms = max_atoms
        self.calls = []

    def evaluate(self, structures, properties):
        self.calls.append(([len(atoms) for atoms in structures], tuple(properties)))
        return [
            {
                "energy": float(atoms.positions.sum()),
                "forces": np.full((len(atoms), 3), float(len(atoms))),
                **({"stress": np.arange(6.0)} if "stress" in properties else {}),
            }
            for atoms in structures
        ]


# Chunking ------------------------------------------------------------------


def _requests(*sizes):
    return [_Request("m", _atoms(size), ("energy", "forces")) for size in sizes]


def test_chunks_without_budget_return_one_chunk():
    requests = _requests(5, 7, 9)
    assert list(_chunks(requests, None)) == [requests]


def test_chunks_respect_the_atom_budget_in_order():
    chunks = list(_chunks(_requests(4, 4, 4, 3, 6), 8))
    assert [[len(r.atoms) for r in chunk] for chunk in chunks] == [[4, 4], [4, 3], [6]]


def test_structure_larger_than_budget_gets_its_own_chunk():
    chunks = list(_chunks(_requests(2, 20, 2), 8))
    assert [[len(r.atoms) for r in chunk] for chunk in chunks] == [[2], [20], [2]]


# Backend selection ---------------------------------------------------------


def test_as_backend_keeps_backends_and_sets_their_budget():
    backend = Recording()
    assert as_backend(backend, max_atoms=100) is backend
    assert backend.max_atoms == 100


def test_as_backend_selects_by_capability():
    class Batched:
        def calculate_batch(self, structures, properties, batch_mode):
            return []

    def function(structures, properties):
        return []

    assert isinstance(as_backend(Batched()), SymmetrixBackend)
    assert isinstance(as_backend(EMT()), SequentialBackend)
    assert isinstance(as_backend(function), FunctionBackend)
    with pytest.raises(TypeError, match="Cannot use"):
        as_backend(42)


def test_symmetrix_backend_requires_a_symmetrix_calculator():
    with pytest.raises(TypeError, match="Symmetrix calculator"):
        SymmetrixBackend(EMT())


def test_symmetrix_backend_forwards_structures_and_mode():
    seen = {}

    class Batched:
        def calculate_batch(self, structures, properties, batch_mode):
            seen.update(count=len(structures), properties=properties, mode=batch_mode)
            return ["result"] * len(structures)

    backend = SymmetrixBackend(Batched(), batch_mode="padded")
    assert backend.evaluate([_atoms(2), _atoms(3)], ["energy", "forces"]) == [
        "result",
        "result",
    ]
    assert seen == {"count": 2, "properties": ("energy", "forces"), "mode": "padded"}


def test_sequential_backend_returns_independent_copies():
    backend = SequentialBackend(EMT())
    first, second = backend.evaluate(
        [bulk("Cu", cubic=True), bulk("Cu", cubic=True)], ("energy", "forces")
    )
    first["forces"][0, 0] = 99.0
    assert second["forces"][0, 0] != 99.0
    assert first["forces"] is not backend.calculator.results["forces"]


# TorchSim backend with a fake torch_sim module -----------------------------


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


@pytest.fixture
def fake_torch_sim(monkeypatch):
    """Install a minimal torch_sim whose state is the list of structures."""

    class ModelInterface:
        compute_stress = True
        device = "cpu"
        dtype = "float64"

        def __call__(self, state):
            energy = [float(i + 1) for i in range(len(state))]
            forces = np.concatenate(
                [np.full((len(atoms), 3), i + 1.0) for i, atoms in enumerate(state)]
            )
            # Symmetric 3x3 stresses with distinct entries per structure.
            stress = [
                np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]], dtype=float) * (i + 1)
                for i in range(len(state))
            ]
            return {
                "energy": _Tensor(energy),
                "forces": _Tensor(forces),
                "stress": _Tensor(stress),
            }

    package = types.ModuleType("torch_sim")
    package.io = types.SimpleNamespace(
        atoms_to_state=lambda structures, device, dtype: list(structures)
    )
    interface = types.ModuleType("torch_sim.models.interface")
    interface.ModelInterface = ModelInterface
    monkeypatch.setitem(sys.modules, "torch_sim", package)
    monkeypatch.setitem(sys.modules, "torch_sim.models.interface", interface)
    return ModelInterface


def test_torch_sim_models_are_detected_once_torch_sim_is_imported(fake_torch_sim):
    assert isinstance(as_backend(fake_torch_sim()), TorchSimBackend)


def test_torch_sim_backend_splits_outputs_per_structure(fake_torch_sim):
    backend = TorchSimBackend(fake_torch_sim())

    results = backend.evaluate([_atoms(2), _atoms(3)], ("energy", "forces", "stress"))

    assert [result["energy"] for result in results] == [1.0, 2.0]
    assert [result["forces"].shape for result in results] == [(2, 3), (3, 3)]
    np.testing.assert_array_equal(results[1]["forces"], np.full((3, 3), 2.0))
    # ASE Voigt order: xx, yy, zz, yz, xz, xy.
    np.testing.assert_array_equal(results[0]["stress"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(results[1]["stress"], [2, 4, 6, 8, 10, 12])


def test_torch_sim_backend_omits_stress_unless_requested(fake_torch_sim):
    (result,) = TorchSimBackend(fake_torch_sim()).evaluate(
        [_atoms(2)], ("energy", "forces")
    )
    assert "stress" not in result


def test_torch_sim_backend_rejects_stress_from_a_model_without_it(fake_torch_sim):
    model = fake_torch_sim()
    model.compute_stress = False
    with pytest.raises(ValueError, match="compute_stress"):
        TorchSimBackend(model).evaluate([_atoms(2)], ("energy", "forces", "stress"))


# Scheduler ------------------------------------------------------------------


def _scheduler(backend):
    scheduler = Scheduler()
    scheduler.add_model("m", backend)
    return scheduler


def test_task_result_reports_success():
    assert TaskResult("a", value=1).ok
    assert not TaskResult("a", error=ValueError()).ok


def test_submit_names_tasks_and_rejects_unknown_models():
    scheduler = _scheduler(Recording())
    assert scheduler.submit(lambda context: None, "m") == "task-0"
    assert scheduler.submit(lambda context: None, "m", name="x") == "x"
    with pytest.raises(KeyError, match="Unknown model"):
        scheduler.submit(lambda context: None, "other")


def test_free_energy_defaults_to_energy():
    scheduler = _scheduler(Recording())
    scheduler.submit(lambda context: context.evaluate([_atoms(2)])[0], "m")

    (result,) = scheduler.run()

    assert result.value["free_energy"] == result.value["energy"]


def test_backend_returning_wrong_count_fails_its_tasks():
    scheduler = _scheduler(FunctionBackend(lambda structures, properties: []))
    scheduler.submit(single_point(_atoms(2)), "m")

    (result,) = scheduler.run()

    assert isinstance(result.error, RuntimeError)
    assert "returned 0 results for 1" in str(result.error)


def test_evaluate_without_structures_makes_no_call():
    backend = Recording()
    scheduler = _scheduler(backend)
    scheduler.submit(lambda context: context.evaluate([]), "m")

    (result,) = scheduler.run()

    assert result.value == []
    assert backend.calls == []


def test_context_rejects_unknown_models():
    scheduler = _scheduler(Recording())
    scheduler.submit(lambda context: context.calculator(model="other"), "m")
    scheduler.submit(lambda context: context.evaluate([_atoms(1)], model="other"), "m")

    results = scheduler.run()

    assert all(isinstance(result.error, KeyError) for result in results)


def test_evaluate_routes_to_the_named_model():
    first, second = Recording(), Recording()
    scheduler = Scheduler()
    scheduler.add_model("first", first)
    scheduler.add_model("second", second)

    def task(context):
        context.evaluate([_atoms(2)])
        context.evaluate([_atoms(3)], model="second")

    scheduler.submit(task, "first")
    scheduler.run()

    assert [sizes for sizes, _ in first.calls] == [[2]]
    assert [sizes for sizes, _ in second.calls] == [[3]]


def test_requests_with_and_without_stress_use_separate_calls():
    backend = Recording()
    scheduler = _scheduler(backend)
    scheduler.submit(single_point(_atoms(2)), "m")
    scheduler.submit(single_point(_atoms(3), ("energy", "forces", "stress")), "m")

    scheduler.run()

    assert sorted(backend.calls) == [
        ([2], ("energy", "forces")),
        ([3], ("energy", "forces", "stress")),
    ]


def test_scheduler_can_run_again_with_new_tasks():
    scheduler = _scheduler(Recording())
    scheduler.submit(lambda context: 1, "m")
    assert [result.value for result in scheduler.run()] == [1]
    scheduler.submit(lambda context: 2, "m")
    assert [result.value for result in scheduler.run()] == [2]


# BatchCalculator caching -----------------------------------------------------


def test_calculator_evaluates_once_per_state():
    backend = Recording()
    scheduler = _scheduler(backend)

    def task(context):
        atoms = _atoms(2)
        atoms.calc = context.calculator()
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.positions[0, 0] += 0.1
        atoms.get_forces()
        atoms.get_potential_energy()

    scheduler.submit(task, "m")
    scheduler.run()

    assert len(backend.calls) == 2


def test_calculator_includes_declared_properties_in_every_request():
    backend = Recording()
    scheduler = _scheduler(backend)

    def task(context):
        atoms = _atoms(2)
        atoms.calc = context.calculator(properties=("energy", "forces", "stress"))
        atoms.get_forces()
        return atoms.get_stress()

    scheduler.submit(task, "m")
    (result,) = scheduler.run()

    assert backend.calls == [([2], ("energy", "forces", "stress"))]
    np.testing.assert_array_equal(result.value, np.arange(6.0))


def test_linked_calculators_resubmit_only_changed_structures():
    backend = Recording()
    scheduler = _scheduler(backend)

    def task(context):
        images = [_atoms(1), _atoms(2), _atoms(3)]
        context.linked_calculators(images)
        for image in images:
            image.get_forces()
        images[1].positions[0, 0] += 0.1
        for image in images:
            image.get_forces()

    scheduler.submit(task, "m")
    scheduler.run()

    assert [sizes for sizes, _ in backend.calls] == [[1, 2, 3], [2]]


def test_requests_are_copied_at_submission():
    seen = []

    def evaluate(structures, properties):
        seen.extend(atoms.positions.copy() for atoms in structures)
        return [{"energy": 0.0, "forces": np.zeros((len(a), 3))} for a in structures]

    scheduler = _scheduler(evaluate)

    def task(context):
        atoms = _atoms(1)
        atoms.calc = context.calculator()
        atoms.get_forces()
        atoms.positions[:] = 99.0
        return atoms

    scheduler.submit(task, "m")
    scheduler.run()

    np.testing.assert_array_equal(seen[0], [[0.0, 1.0, 2.0]])


# Swap Monte Carlo proposals --------------------------------------------------


def test_swap_monte_carlo_moves_only_the_given_sites():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    atoms.numbers[[0, 5]] = 79
    sites = [0, 1, 2, 3]
    scheduler = _scheduler(
        lambda structures, properties: [
            {"energy": 0.0, "forces": np.zeros((len(a), 3))} for a in structures
        ]
    )
    scheduler.submit(swap_monte_carlo(atoms, [300.0], 50, sites=sites, rng=0), "m")

    (result,) = scheduler.run()

    final = result.value["replicas"][0]
    np.testing.assert_array_equal(final.numbers[4:], atoms.numbers[4:])
    assert sorted(final.numbers[:4]) == sorted(atoms.numbers[:4])
    # Flat energy surface: every proposal is accepted.
    assert result.value["acceptance"][0] == 1.0


def test_swap_monte_carlo_proposes_only_unlike_pairs():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    atoms.numbers[0] = 79
    scheduler = _scheduler(
        lambda structures, properties: [
            {"energy": 0.0, "forces": np.zeros((len(a), 3))} for a in structures
        ]
    )
    for seed in range(20):
        scheduler.submit(swap_monte_carlo(atoms, [300.0], 1, rng=seed), "m")

    results = scheduler.run()

    # A like pair would leave the configuration unchanged after one step.
    for result in results:
        assert result.value["replicas"][0].numbers[0] != 79


# Symmetrix neighbor lists ----------------------------------------------------


@pytest.fixture
def fake_symmetrix(monkeypatch):
    """Install a symmetrix package whose calculator module has a neighbor list."""

    def install(neighbor_list):
        package = types.ModuleType("symmetrix")
        module = types.ModuleType("symmetrix.calculator")
        module.neighbor_list = neighbor_list
        package.calculator = module
        monkeypatch.setitem(sys.modules, "symmetrix", package)
        monkeypatch.setitem(sys.modules, "symmetrix.calculator", module)

    return install


class _Calculator:
    cutoff = 5.0

    evaluator = types.SimpleNamespace(atomic_numbers=(8, 22, 38))

    def _compute_mace(self, mace_inputs, atoms=None):
        raise NotImplementedError


def test_graph_batcher_warns_about_the_ase_neighbor_list(fake_symmetrix):
    from ase.neighborlist import neighbor_list

    from atombatch.symmetrix import GraphBatcher

    fake_symmetrix(neighbor_list)
    with pytest.warns(UserWarning, match="matscipy"):
        GraphBatcher(_Calculator())


def test_graph_batcher_is_silent_with_a_compiled_neighbor_list(fake_symmetrix):
    import warnings

    from atombatch.symmetrix import GraphBatcher

    def neighbor_list(quantities, atoms, cutoff):
        return ()

    neighbor_list.__module__ = "matscipy.neighbours"
    fake_symmetrix(neighbor_list)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GraphBatcher(_Calculator())


# BatchCalculator.check_state -------------------------------------------------


def _changed_pairs():
    base = bulk("Cu", "fcc", a=3.6, cubic=True)
    base.set_initial_magnetic_moments([1.0, 0.0, 0.0, 0.0])

    def variant(change):
        atoms = base.copy()
        change(atoms)
        return atoms

    def shift(delta):
        def change(atoms):
            atoms.positions[0, 0] += delta

        return change

    return {
        "identical": variant(lambda atoms: None),
        "within tolerance": variant(shift(5e-16)),
        "moved": variant(shift(1e-6)),
        "cell": variant(lambda atoms: atoms.set_cell(atoms.cell * 1.01)),
        "pbc": variant(lambda atoms: atoms.set_pbc([True, True, False])),
        "numbers": variant(lambda atoms: atoms.numbers.__setitem__(0, 79)),
        "magmoms": variant(lambda atoms: atoms.set_initial_magnetic_moments(None)),
        "charges": variant(lambda atoms: atoms.set_initial_charges([1, 0, 0, 0])),
    }, base


@pytest.mark.parametrize("ignored", [(), ("initial_magmoms",)])
def test_check_state_matches_ase(ignored):
    from ase.calculators.calculator import compare_atoms

    from atombatch import BatchCalculator

    variants, base = _changed_pairs()
    calculator = BatchCalculator(None, "m")
    calculator.atoms = base.copy()
    calculator.ignored_changes = set(ignored)
    for name, atoms in variants.items():
        expected = compare_atoms(
            base, atoms, tol=1e-15, excluded_properties=set(ignored)
        )
        assert sorted(calculator.check_state(atoms)) == sorted(expected), name


# Edge geometry kernel ----------------------------------------------------------


def _random_edges(seed, count=5000, atoms=40):
    rng = np.random.default_rng(seed)
    reference = rng.normal(scale=3.0, size=(count, 3))
    sources = rng.integers(0, atoms, count).astype(np.int32)
    receivers = np.sort(rng.integers(0, atoms, count)).astype(np.int32)
    displacements = rng.normal(scale=0.1, size=(atoms, 3))
    return reference, sources, receivers, displacements


@pytest.mark.parametrize("clamp", [False, True])
def test_compiled_edge_geometry_matches_numpy_bit_for_bit(clamp):
    from atombatch import _edges

    if _edges.numba is None:
        pytest.skip("numba is not installed")
    for seed in range(3):
        arguments = (*_random_edges(seed), 5.0, clamp)
        compiled = _edges.edge_geometry(*arguments)
        reference = _edges.edge_geometry_numpy(*arguments)
        for got, want in zip(compiled, reference):
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)


def test_edge_geometry_falls_back_to_numpy(monkeypatch):
    from atombatch import _edges

    monkeypatch.setattr(_edges, "numba", None)
    arguments = (*_random_edges(0), 5.0, False)
    for got, want in zip(
        _edges.edge_geometry(*arguments), _edges.edge_geometry_numpy(*arguments)
    ):
        np.testing.assert_array_equal(got, want)
