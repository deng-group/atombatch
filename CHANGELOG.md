# Changelog

## 0.1.3 (2026-10-17)

Large structures with Symmetrix now run as fast through atombatch as with the Symmetrix calculator alone (issue #1), and atombatch gains random structure searching and an atomate2 integration.

- Large structures (1000 atoms or more by default, `SymmetrixBackend(native_atoms=...)`) are evaluated one at a time on Symmetrix's prepared-geometry path, which keeps the neighbor graph on the device and reduces forces and stress there, instead of in the joined graph, whose host-side work grows with the number of edges. `SymmetrixBackend.from_model(model_file, **options)` also gives up to `max_calculators` (4) large trajectories a calculator of their own, so that Symmetrix can build their neighbor graphs on the device; calculators return to a pool when their task finishes. On an RTX 4090 (FP32), one MD task of 10,648 Si atoms with energy, forces, and stress costs 1.00 times the Symmetrix calculator (4.2 times in 0.1.2), and three such structures per call cost 76.6 ms against 77.4 ms one after another (290 ms in 0.1.2). Each extra calculator costs one model load and about 0.8 GB of device memory at that size (see the performance page).
- `airss` and `hot_airss` run random structure searches, one task per sample, at a given pressure: a random structure relaxed in positions and cell, and for hot-AIRSS a Langevin anneal and a second relaxation. Random structures come from the AIRSS `buildcell` program (`atombatch.workflows.airss.buildcell`, with a seed file or an airsspy `SeedAtoms`) or from a built-in generator without symmetry (`random_structure`); `rank` sorts and merges the minima. `examples/airss_carbon.py` finds diamond, lonsdaleite, and graphite-like carbon at 20 GPa.
- `atombatch.integrations.atomate2.atomate2_task` runs atomate2 makers that go through ASE (`ForceFieldRelaxMaker`, `ForceFieldStaticMaker`, `AseRelaxMaker` subclasses) with batched evaluations and returns the documents the jobs would return. 28 Symmetrix cell relaxations ran 2.4 times faster than the same jobs one after another on an M1 Max. New `atomate2` extra (Python 3.11 or later).
- `molecular_dynamics` passes the moving `Atoms` to observers as a positional argument; observers with a parameter not named `atoms` failed.
- Requests identify their trajectory with a `Source` object, which the scheduler drops when a task finishes, so backends release per-trajectory state at once.
- The package is organized into `atombatch.backends` (base, Symmetrix, TorchSim) and `atombatch.workflows` (relax, NEB, MD, EOS, phonons, Monte Carlo, AIRSS). `from atombatch import ...` is unchanged, and `atombatch.tasks` and `atombatch.symmetrix` still re-export their 0.1.2 contents.
- Examples for every workflow (cell relaxations of all orderings, NEB, MD, phonons, swap Monte Carlo, AIRSS, atomate2) and a SLURM job template, in `examples/` and on the Examples page of the documentation.

## 0.1.2 (2026-10-15)

Faster batched relaxations, tested on CUDA. On an NVIDIA RTX 4090, 64 FIRE relaxations of 40-atom SrTiO3 cells with Symmetrix (FP32) run 4.2 times faster than a sequential ASE loop (1.5 times in 0.1.1), and 4.8 times on an Apple M1 Max, with identical step counts. Results remain identical to independent ASE runs.

- `BatchCalculator.check_state` compares arrays for exact equality before ASE's `np.allclose` tolerance, with the same result as ASE. ASE calls it about four times per optimizer step; overhead per evaluation fell from about 260 to 100 us.
- Batched Symmetrix evaluation keeps one Symmetrix neighbor cache per trajectory, so neighbor lists are rebuilt only when atoms move beyond the neighbor skin, as in the Symmetrix ASE calculator. Requests carry the identity of their calculator in `Atoms.info["atombatch_source"]`. The graph path is used even when `calculate_batch` is available.
- Cached neighbor edges are updated by one loop compiled with Numba (`atombatch.backends._edges`), identical bit for bit to Symmetrix's own (74 instead of 200 us per structure). The `symmetrix` extra installs Numba; without it, a NumPy version with the same results is used.
- The batched graph is passed to Symmetrix as `int32` arrays instead of Python lists, which were converted element by element.
- The check for Symmetrix's slow ASE neighbor-list fallback also works with the neighbor-list wrapper of symmetrix-xl PR #2.
- Tested on an RTX 4090 with the Symmetrix-XL 0.1.1 CUDA wheel and with MACE through TorchSim. The installation guide documents two CUDA environment problems found there: the Symmetrix CUDA wheels loading an older `libnvrtc.so` or `libcudart.so`, and Triton needing Python development headers.
- README restructured around installation and a Symmetrix quick start. `ATOMBATCH_MACE_DEVICE` selects the device of the MACE test, and the benchmark script takes `--dtype`.

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
