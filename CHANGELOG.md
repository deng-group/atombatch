# Changelog

## 0.1.1 (2026-10-15)

- The `symmetrix` extra installs matscipy. Without it, Symmetrix uses ASE's Python neighbor list, and batched relaxations with the `metal-preview-1` wheel ran three times slower than a sequential loop. atombatch now warns when Symmetrix has fallen back to the ASE neighbor list.
- Tested with the `metal-preview-1` Symmetrix-XL wheels (Metal, no `calculate_batch`): batched results match the Symmetrix calculator, and 64 FIRE relaxations ran 2.0 times faster than a sequential loop with identical step counts.
- Releases are published to PyPI from GitHub Actions with trusted publishing.

## 0.1.0 (2026-10-15)

First public release.

- Scheduler that runs unmodified ASE tasks in worker threads and evaluates their pending energy and force requests in one batched call per model, with refill from a queue (`max_active`) and a per-model atom budget (`max_atoms`).
- Tasks: single points, relaxations with any ASE optimizer and cell filter, NEB (regular and climbing image), and molecular dynamics with any ASE integrator.
- Batched workflows: `TaskContext.evaluate` for independent structures, force sets for phonopy and phono3py (`displacement_forces`), equations of state (`equation_of_state`), and swap Monte Carlo with replica exchange (`swap_monte_carlo`).
- Backends: Symmetrix-XL from 0.1.1 (native `calculate_batch` when available, otherwise one joined neighbor graph per call), TorchSim models (MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, metatomic), user functions, and any ASE calculator evaluated sequentially.
- Results agree with independent ASE runs and with the models' own ASE calculators to rounding.
- Benchmarks: on an Apple M1 Max GPU with Symmetrix, 6.2 times higher model throughput at 64 structures per call and 3.0 times faster FIRE relaxations than a sequential ASE loop.
- Documentation (Sphinx), unit and integration tests, and a benchmark script.
