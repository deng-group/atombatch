# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` for the user-facing description and `docs/design.md` for the design decisions behind the code.

## Project

atombatch batches the energy and force requests of many independent ASE tasks (single points, relaxations, NEB, molecular dynamics) into one model call per model. The ASE optimizers and integrators run unmodified, and every task receives exactly the result it would get from an independent run. Preserving that equivalence is the central invariant of the project.

## Layout

- `src/atombatch/scheduler.py`: `Scheduler`, `BatchCalculator`, `TaskContext`, `TaskResult`. Each task runs in its own worker thread. `BatchCalculator.calculate` submits a request and blocks. The main thread waits until every active task is blocked or finished, then `_serve` groups requests by `(model, needs stress)`, splits them into chunks under the backend's `max_atoms`, evaluates each chunk, and wakes the tasks. Finished tasks are replaced from the queue by `_admit`, bounded by `max_active`.
- `src/atombatch/backends.py`: `Backend` interface (`evaluate(structures, properties) -> list[dict]`), `SequentialBackend` for any ASE calculator, `SymmetrixBackend`, which uses `Symmetrix.calculate_batch` when the installed Symmetrix has it and otherwise the graph batching in `src/atombatch/symmetrix.py` (adapted from Symmetrix-XL, MIT, and limited to evaluator calls present in Symmetrix-XL 0.1.1), `TorchSimBackend` wrapping a TorchSim `ModelInterface` (MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, metatomic), and `FunctionBackend` for a user function with the `evaluate` signature. `as_backend` picks the backend by type and duck typing. It recognizes TorchSim models only when `torch_sim` is already imported, so ASE-only runs never import torch.
- `src/atombatch/_edges.py`: per-step edge geometry of a reused Symmetrix neighbor cache, compiled with Numba when available. It must stay bit-identical to `Symmetrix._cached_neighbor_geometry`: same operations in the same order, no `fastmath`. `tests/test_units.py` and `tests/test_symmetrix_backend.py` check this.
- `src/atombatch/tasks.py`: task factories `single_point`, `relax`, `neb`, `molecular_dynamics`, and the batched workflows `displacement_forces` (phonopy and phono3py), `equation_of_state`, and `swap_monte_carlo`. Each returns a closure `task(context)` that copies its input structures and runs plain ASE code with calculators from the context. Workflows with independent evaluations submit them together through `TaskContext.evaluate`.
- `examples/`: runnable Symmetrix scripts for each workflow, with `README.md` (listed on the docs Examples page) and a SLURM template. They need a Symmetrix model in `SYMMETRIX_MODEL` and are not run by CI; run them after API changes and update the quoted results and times.
- `tests/test_units.py`: fast unit tests of chunking, backend selection (with a fake `torch_sim` module), result splitting, calculator caching, linked calculators, error paths, and Monte Carlo proposals. They use small fake backends and run in under a second.
- `tests/test_scheduler.py`: tests against EMT, which needs no MLIP. `tests/test_workflows.py` covers `context.evaluate` and the batched workflows (phonopy and phono3py tests are skipped when those packages are absent). `tests/test_torch_sim_backend.py` uses TorchSim's Lennard-Jones model and is skipped when TorchSim is not installed. `tests/test_mace_backend.py` compares batched MACE through TorchSim with MACE's own ASE calculator and runs only when `ATOMBATCH_MACE_MODEL` names a MACE-MP model (for example `small`) and `mace-torch` is installed. `tests/test_symmetrix_backend.py` is skipped unless `ATOMBATCH_SYMMETRIX_MODEL` points to a Symmetrix JSON model and the installed Symmetrix provides `calculate_batch`.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"          # add ,torchsim on Python 3.12+ for the TorchSim tests
pytest
```

Python 3.10 or later, with `numpy` and `ase>=3.23` as the only runtime dependencies. Symmetrix (`.[symmetrix]`) and TorchSim (`.[torchsim]`, Python 3.12 or later) are optional extras and must stay optional: import them inside the backend, never at module level. Ruff is configured with line length 88 in `pyproject.toml`, and existing `noqa` comments carry the reason in parentheses. Follow that convention.

## Documentation

Sphinx with the Furo theme and MyST Markdown, in `docs/` (`conf.py`, guide pages in Markdown, `api.rst` for autodoc). `docs/design.md` is part of the site. Build with `sphinx-build -W docs docs/_build/html`, which must pass without warnings. `.github/workflows/docs.yml` publishes it to GitHub Pages (https://deng-group.github.io/atombatch/) on every push to `main`. Code examples in `index.md`, `quickstart.md`, `workflows.md`, and `custom-tasks.md` use EMT so that they run without an MLIP. Run them after changing the API, and keep any printed output quoted in the pages in sync. Docstrings use Sphinx roles (`:class:`, `:meth:`) and double-backtick literals. The logo is `docs/_static/logo.svg`, a flat 3x3 grid of atoms in indigo tones with one amber atom.

## Conventions

- Validate every new task type or scheduling change by comparing with an independent ASE run of the same algorithm on EMT: identical step counts and convergence, and positions, energies, and forces equal within `atol=1e-9` (see `assert_same` in `tests/test_scheduler.py`). The tolerance is nonzero only because EMT reuses its neighbor list in an independent run.
- A request must be evaluated at exactly the state the task asked for. `_evaluate` copies `atoms` at submission time, so later mutation by the task cannot leak into the batch.
- A workflow whose evaluations do not depend on each other must submit them through `context.evaluate` in one request rather than one at a time, otherwise it contributes one structure per round.
- Tasks that query several structures in turn, such as an NEB band, use `TaskContext.linked_calculators` so that one request covers every stale member of the group. Tasks that read stress (cell filters) pass `properties` including `"stress"` so that each state is evaluated once.
- Exceptions raised in a task are captured in its `TaskResult.error` and do not stop other tasks. A backend failure is raised in every task of the failing chunk.
- New MLIP backends are added one at a time, each validated against that model's own ASE calculator. A backend returns one dict per structure with `energy`, `forces`, and `stress` in ASE Voigt order when requested.
- Public API changes must be reflected in `src/atombatch/__init__.py` (`__all__`), the README example, and `docs/design.md` where the design changes.

## Status

Early development (version 0.1.2). The version is set in both `pyproject.toml` and `src/atombatch/__init__.py`, and `CHANGELOG.md` records each release. Correctness is tested. `benchmarks/bench_batching.py` measures model throughput against batch size and end-to-end relaxations against a sequential ASE loop, and its results are quoted in `docs/performance.md` with the JSON in `benchmarks/results/`. When quoting new numbers, state the hardware, model, precision, structure size, and `max_active`, and repeat CPU runs, which vary by about 20 %.
