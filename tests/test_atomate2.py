import sys

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from atombatch import Scheduler, SequentialBackend
from atombatch.integrations.atomate2 import atomate2_task

pytest.importorskip("atomate2")
from atomate2.ase.jobs import EmtRelaxMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.io.ase import AseAtomsAdaptor


class EmtForceFieldRelaxMaker(ForceFieldRelaxMaker):
    """A force-field maker whose calculator is EMT, for comparison."""

    def _get_calculator(self):
        return EMT()


def _structures(count=4):
    structures = []
    for seed in range(count):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
        atoms.numbers[np.random.default_rng(seed).choice(32, 4, replace=False)] = 79
        atoms.rattle(0.05, seed=seed)
        structures.append(AseAtomsAdaptor.get_structure(atoms))
    return structures


def _sequential(maker, structure):
    return type(maker).make.original(maker, structure)


def _requested_only(structures, properties):
    # EMT computes stress whether or not it is requested; batched models such
    # as Symmetrix and TorchSim return only the requested properties.
    results = SequentialBackend(EMT()).evaluate(structures, properties)
    return [{key: result[key] for key in properties} for result in results]


def _run_batched(maker, structures):
    scheduler = Scheduler()
    scheduler.add_model("emt", _requested_only)
    for structure in structures:
        scheduler.submit(atomate2_task(maker, structure), "emt")
    results = scheduler.run()
    assert all(result.ok for result in results), [r.error for r in results]
    return scheduler, [result.value for result in results]


@pytest.mark.parametrize(
    "maker",
    [
        EmtRelaxMaker(relax_kwargs={"fmax": 0.02}),
        EmtForceFieldRelaxMaker(relax_cell=True, relax_kwargs={"fmax": 0.02}),
        EmtForceFieldRelaxMaker(relax_cell=False, relax_kwargs={"fmax": 0.02}),
    ],
    ids=["ase", "forcefield-cell", "forcefield-fixed-cell"],
)
def test_batched_atomate2_documents_match_sequential_jobs(maker):
    structures = _structures()

    scheduler, documents = _run_batched(maker, structures)

    # The relaxations of all structures share the first call.
    assert scheduler.batch_log[0][1] == len(structures)
    for document, structure in zip(documents, structures):
        reference = _sequential(maker, structure)
        assert type(document) is type(reference)
        assert document.output.energy == pytest.approx(
            reference.output.energy, abs=1e-9
        )
        np.testing.assert_allclose(
            document.output.structure.cart_coords,
            reference.output.structure.cart_coords,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            document.output.structure.lattice.matrix,
            reference.output.structure.lattice.matrix,
            atol=1e-8,
        )
        assert document.output.n_steps == reference.output.n_steps


@pytest.mark.parametrize("relax_cell", [True, False])
def test_each_recorded_state_is_evaluated_once(relax_cell):
    # atomate2 reads stress of periodic structures at every step, also when
    # the cell is fixed, so it must come with the forces.
    maker = EmtForceFieldRelaxMaker(relax_cell=relax_cell, relax_kwargs={"fmax": 0.02})
    structures = _structures(2)

    scheduler, documents = _run_batched(maker, structures)

    evaluations = sum(size for _, size, _ in scheduler.batch_log)
    # Forces and stress of each recorded state come from one evaluation;
    # atomate2 records the final state once more, from the cache.
    recorded = sum(len(doc.output.ionic_steps) for doc in documents)
    assert evaluations == recorded - len(structures)


def test_non_ase_makers_are_rejected():
    class NotAMaker:
        def make(self, structure):
            return structure

    with pytest.raises(TypeError, match="atomate2 maker"):
        atomate2_task(NotAMaker(), _structures(1)[0])


def test_concurrent_jobs_leave_stdout_intact():
    # atomate2 redirects sys.stdout around each relaxation; concurrent
    # relaxations must not leave it pointing at another job's buffer.
    maker = EmtForceFieldRelaxMaker(relax_cell=True, relax_kwargs={"fmax": 0.02})
    stdout = sys.stdout
    for _ in range(3):
        _, documents = _run_batched(maker, _structures(6))
        assert sys.stdout is stdout
    assert documents[0].input.relax_kwargs == {"fmax": 0.02}
