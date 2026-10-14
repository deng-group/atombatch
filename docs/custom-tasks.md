# Writing tasks

A task is any function that takes a {class}`~atombatch.TaskContext` and returns a value. The scheduler runs it in its own thread, and the return value becomes {attr}`TaskResult.value <atombatch.TaskResult.value>`. The built-in tasks are written this way, so any ASE script can become a task.

The context provides three ways to request evaluations, depending on how the script uses the model.

## One structure: a batched calculator

{meth}`context.calculator() <atombatch.TaskContext.calculator>` returns an ASE calculator. Attach it to an `Atoms` object and use any ASE code. Each evaluation blocks until the scheduler has evaluated it together with the requests of the other tasks.

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from atombatch import Scheduler


def relax_then_strain(context):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=0)
    atoms.calc = context.calculator(properties=("energy", "forces", "stress"))
    BFGS(atoms, logfile=None).run(fmax=0.05)
    if atoms.get_stress()[0] > 0:
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
        BFGS(atoms, logfile=None).run(fmax=0.05)
    return atoms.get_potential_energy()


scheduler = Scheduler()
scheduler.add_model("emt", EMT())
scheduler.submit(relax_then_strain, "emt")
print(scheduler.run()[0].value)
```

Pass every property the script reads at each state in `properties`. A script that reads forces and stress at the same positions, as a cell filter does, is then evaluated once per state rather than twice.

## Independent structures: `context.evaluate`

When a task needs many evaluations that do not depend on each other, submit them in one request. {meth}`context.evaluate(structures) <atombatch.TaskContext.evaluate>` returns one result dictionary per structure, with `energy`, `free_energy`, `forces`, and any further requested properties.

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from atombatch import Scheduler


def vacancy_energies(context):
    host = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3)
    vacancies = []
    for index in range(4):
        cell = host.copy()
        del cell[index]
        vacancies.append(cell)
    results = context.evaluate([host, *vacancies])
    bulk_energy = results[0]["energy"] / len(host)
    return [r["energy"] - (len(host) - 1) * bulk_energy for r in results[1:]]


scheduler = Scheduler()
scheduler.add_model("emt", EMT())
scheduler.submit(vacancy_energies, "emt")
print(scheduler.run()[0].value)
```

Evaluating these structures one at a time through a calculator would take five rounds. With `context.evaluate` they take one.

## Structures queried in turn: linked calculators

Some algorithms query several structures one after another within a single step. NEB, for example, reads the forces of each image in turn. {meth}`context.linked_calculators(structures) <atombatch.TaskContext.linked_calculators>` attaches one calculator to each structure and groups them: the first query submits every structure in the group whose state changed, and the following queries are served from the cache. {func}`~atombatch.neb` is built this way.

## Several models in one task

`context.calculator`, `context.evaluate`, and `context.linked_calculators` each take an optional `model` name, which defaults to the model the task was submitted with. A task can therefore compare models, or screen with a cheap model and refine with an expensive one.

## Errors

An exception raised in a task is stored in {attr}`TaskResult.error <atombatch.TaskResult.error>` and does not stop other tasks. If a backend call fails, the exception is raised in every task whose requests were in that call.
