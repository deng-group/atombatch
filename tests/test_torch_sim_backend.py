import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from atombatch import (
    FunctionBackend,
    Scheduler,
    SequentialBackend,
    TorchSimBackend,
    relax,
    single_point,
)

LJ = {"sigma": 2.3, "epsilon": 0.4, "cutoff": 5.0}


@pytest.fixture(scope="module")
def lennard_jones():
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torch_sim.models.lennard_jones")
    return models.LennardJonesModel(
        **LJ, compute_stress=True, dtype=torch.float64, device=torch.device("cpu")
    )


def _ase_lennard_jones():
    # ASE shifts the pair energy to zero at the cutoff and TorchSim does not,
    # so only forces and stress are compared with ASE.
    return LennardJones(
        sigma=LJ["sigma"], epsilon=LJ["epsilon"], rc=LJ["cutoff"], smooth=False
    )


def _structures():
    structures = []
    for index, repeat in enumerate([(2, 2, 2), (2, 2, 3), (1, 2, 2)]):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(repeat)
        atoms.rattle(0.05, seed=index)
        structures.append(atoms)
    return structures


def test_torch_sim_model_is_batched_and_split_per_structure(lennard_jones):
    structures = _structures()
    scheduler = Scheduler()
    scheduler.add_model("lj", lennard_jones)
    assert isinstance(scheduler._backends["lj"], TorchSimBackend)
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "lj")

    results = scheduler.run()

    assert scheduler.batch_log == [("lj", 3, sum(len(a) for a in structures))]
    alone = TorchSimBackend(lennard_jones)
    for result, atoms in zip(results, structures):
        expected = alone.evaluate([atoms], ("energy", "forces", "stress"))[0]
        np.testing.assert_allclose(result.value["energy"], expected["energy"])
        reference = atoms.copy()
        reference.calc = _ase_lennard_jones()
        np.testing.assert_allclose(
            result.value["forces"], reference.get_forces(), atol=1e-10
        )
        np.testing.assert_allclose(
            result.value["stress"], reference.get_stress(), atol=1e-10
        )


def test_relaxations_with_torch_sim_match_sequential_runs(lennard_jones):
    structures = _structures()
    batched = Scheduler()
    batched.add_model("lj", lennard_jones)
    sequential = Scheduler(max_active=1)
    sequential.add_model("lj", lennard_jones)
    for scheduler in (batched, sequential):
        for atoms in structures:
            scheduler.submit(
                relax(atoms, steps=100, cell_filter=FrechetCellFilter), "lj"
            )

    batched_results = batched.run()
    sequential_results = sequential.run()

    assert batched.batch_log[0][1] == len(structures)
    for result, expected in zip(batched_results, sequential_results):
        assert result.value["steps"] == expected.value["steps"]
        np.testing.assert_allclose(
            result.value["atoms"].positions,
            expected.value["atoms"].positions,
            rtol=0.0,
            atol=1e-9,
        )


def test_stress_requires_a_stress_model():
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torch_sim.models.lennard_jones")
    model = models.LennardJonesModel(**LJ, dtype=torch.float64)
    scheduler = Scheduler()
    scheduler.add_model("lj", model)
    scheduler.submit(
        single_point(_structures()[0], ("energy", "forces", "stress")), "lj"
    )

    (result,) = scheduler.run()

    assert isinstance(result.error, ValueError)


def test_function_backend_wraps_a_plain_function():
    sequential = SequentialBackend(EMT())
    calls = []

    def evaluate(structures, properties):
        calls.append(len(structures))
        return sequential.evaluate(structures, properties)

    scheduler = Scheduler()
    scheduler.add_model("emt", evaluate)
    assert isinstance(scheduler._backends["emt"], FunctionBackend)
    for atoms in _structures():
        scheduler.submit(relax(atoms, steps=20, optimizer=FIRE), "emt")

    results = scheduler.run()

    assert all(result.ok for result in results)
    assert calls[0] == 3
