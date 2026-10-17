# Performance

This page reports measurements and explains what determines the cost of a run, so that the settings can be chosen and measured for a given workload.

## Measurements

All runs used rattled SrTiO{sub}`3` supercells of 40 atoms, with `benchmarks/bench_batching.py` (see `benchmarks/README.md`). The sequential baseline evaluates one structure per call through the model's own ASE calculator. The relaxation benchmark runs 64 FIRE relaxations (`fmax=0.05`), once as a plain ASE loop and once through atombatch. Every batched run took the same number of optimizer steps as the sequential run. The GPU rows were measured with atombatch 0.1.2 (faster `check_state`, per-trajectory Symmetrix neighbor lists, and a compiled edge-geometry update, see below), and the CPU rows with 0.1.1.

### Apple M1 Max (10 CPU cores, 32-core GPU)

Model throughput, in us/atom:

| Structures per call | 1 | 4 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| Symmetrix, Metal GPU (FP32), sequential 147 | 142 | 46 | 28 | 25 | 23 | 43 |
| Symmetrix, CPU (FP64, one core), sequential 1510 | 1500 | 1730 | 1110 | 1660 | 1280 | 1220 |
| MACE-MP small via TorchSim, CPU (FP64, 8 threads), sequential 1600 | 1770 | 1460 | 1000 | 1040 | 2950 | 4820 |

Symmetrix ran MACE-OMat-0 medium, built from a Symmetrix-XL source checkout after 0.1.1 with the Apple GPU backend. atombatch's graph path for 0.1.1 gave the same throughput on that build (23.7 against 23.1 us/atom at 64 structures, in an earlier session). The laptop GPU's speed varied between sessions (sequential 145 to 198 us/atom for the same evaluation), so compare speedups within a row rather than absolute values between sessions. With the `metal-preview-1` Symmetrix wheel, which has Metal but no `calculate_batch` and therefore uses atombatch's graph path, throughput was 40 us/atom at 64 structures per call and the 64 relaxations ran 2.0 times faster than the sequential loop (10.6 against 20.9 s, identical step counts). The CPU rows vary by about 20 % between repeated runs, so differences of that size are noise.

Relaxations of 64 cells:

| Model and device | `max_active` | Mean batch | Sequential | atombatch (model + other) | Speedup |
| --- | --- | --- | --- | --- | --- |
| Symmetrix, Metal GPU | 64 | 44 | 17.0 s | 3.5 s (3.21 + 0.31) | 4.8x |
| Symmetrix, CPU | 64 | 44 | 162 s | 178 s (176.7 + 1.1) | 0.9x |
| MACE via TorchSim, CPU | 64 | 47 | 185 s | 268 s (266.5 + 1.5) | 0.7x |
| MACE via TorchSim, CPU | 16 | 15 | 226 s | 125 s (123.9 + 1.0) | 1.8x |

### NVIDIA RTX 4090 (24 GB, Intel Core i9-13900KF host)

With the Symmetrix-XL 0.1.1 CUDA wheel (`symmetrix-xl-cuda12-sm89`, batched through atombatch's graph path) and MACE-MP small through TorchSim 0.6.2 with torch 2.14 (CUDA 13.0):

| Model and precision | Sequential | 4 | 16 | 64 | 128 | Gain at 64 |
| --- | --- | --- | --- | --- | --- | --- |
| Symmetrix FP32 | 62 | 13 | 5.6 | 4.3 | 15 | 14.3x |
| Symmetrix FP64 | 132 | 42 | 27 | 25 | 37 | 5.4x |
| MACE via TorchSim FP32 | 204 | 55 | 46 | 44 | 44 | 4.6x |
| MACE via TorchSim FP64 | 298 | 209 | 193 | 191 | out of memory | 1.6x |

Model throughput in us/atom against structures per call. Relaxations of 64 cells with `max_active=64`:

| Model and precision | Sequential | atombatch (model + other) | Speedup |
| --- | --- | --- | --- |
| Symmetrix FP32 | 6.5 s | 1.6 s (0.9 + 0.6) | 4.2x |
| Symmetrix FP64 | 14.3 s | 3.9 s (3.2 + 0.6) | 3.7x |
| MACE via TorchSim FP32 | 23.4 s | 5.8 s (5.1 + 0.7) | 4.1x |
| MACE via TorchSim FP64 | 34.2 s | 23.1 s (22.4 + 0.7) | 1.5x |

All runs took the same optimizer steps as the sequential loop. FP64 is slow on this consumer GPU, which leaves MACE compute-bound and gains little from batching. With atombatch 0.1.1, the time outside the model call was 1.4 to 2.0 s in these runs; with Symmetrix in FP32 it was 43 % of the run and limited the relaxation speedup to 1.5x. The faster `check_state` reduced it to about 0.65 s (0.23 ms per evaluation) and raised that speedup to 2.4x.

Three conclusions follow from these numbers.

- On the GPUs, batching reduced the cost per atom 5 to 14 times at 64 structures per call (FP32 Symmetrix and MACE, and FP64 Symmetrix) and made the relaxations 3.7 to 4.8 times faster. The relaxation gain is smaller than the throughput gain because relaxations finish at different times, so the mean batch (44) is below `max_active` and the last rounds are nearly empty. Throughput fell again at 128 structures per call, so larger is not always better.
- The time outside the model call is now 0.1 to 0.25 ms per evaluation, 6 to 22 % of the GPU runs. Most of it is ASE's own per-step work (see below), and on fast GPUs it still sets the limit for small cells.
- On the CPU, the gain depends on the model and on the batch size. The single-core Symmetrix build is compute-bound and gains nothing. MACE through TorchSim with 8 threads gained about 1.6 times at 16 to 32 structures per call and lost at 64 and above, so `max_active=16` gave a 1.8x speedup where `max_active=64` gave a slowdown. On a CPU, measure before choosing `max_active`.

### Large structures with Symmetrix

Joining structures into one graph pays off for small cells, but for large ones the host-side work of the joined graph grows with the number of edges: the edge vectors are built on the host and uploaded, and the stresses are reduced on the host from the per-edge forces, which Symmetrix-XL 0.1.1 has to copy back from the device. Symmetrix's own calculator instead keeps the neighbor graph on the device and uploads only the positions. Structures with at least 1000 atoms (`native_atoms` of {class}`~atombatch.SymmetrixBackend`) are therefore evaluated one at a time on Symmetrix's prepared-geometry path, with the neighbor cache of their trajectory, and their results equal the Symmetrix calculator's bit for bit.

The threshold comes from evaluating Si supercells with energy, forces, and stress on an RTX 4090 (FP32, with small moves between calls as in a relaxation; the GPU was shared with another job, so the ratios are the reliable part):

| Atoms per structure | Structures | Joined graph | One prepared call each |
| --- | --- | --- | --- |
| 64 | 32 | 35 ms | 284 ms |
| 512 | 16 | 114 ms | 165 ms |
| 1000 | 8 | 108 ms | 111 ms |
| 1728 | 8 | 196 ms | 147 ms |
| 10648 | 3 | 485 ms | 182 ms |

Structures sharing the calculator must use host-built neighbor graphs: a device-resident graph is identified inside the evaluator and cannot be exchanged between trajectories. For large structures the Symmetrix calculator alone builds the graph on the device, which is about 10 % faster (47.4 against 52.6 ms per call for 10,648 Si atoms). {meth}`SymmetrixBackend.from_model <atombatch.SymmetrixBackend.from_model>` therefore creates, besides the shared calculator, up to `max_calculators` (4) calculators of their own for trajectories of large structures, which then run exactly as with the Symmetrix calculator alone. A calculator returns to a pool when its task finishes and is reused by the next trajectory; trajectories beyond the limit share the main calculator. Each costs one model load (7 s for the 292 MB MACE-MP-0b medium export) and its own device memory (about 0.8 GB for 10,648 atoms).

Issue #1 measured one MD trajectory with energy, forces, and stress at every step, and three large structures per call. On an idle RTX 4090 (FP32), relative to the Symmetrix calculator alone:

| Case | atombatch 0.1.2 | Shared calculator | `from_model` |
| --- | --- | --- | --- |
| One MD task, 1000 Si atoms | 2.4x | 1.07x | 0.94x |
| One MD task, 10,648 Si atoms | 4.2x | 1.13x | 0.99x |
| 3 x 10,648 atoms in one call, against one after another | 3.6x | 1.14x | 0.99x |

For large structures in long runs, create the backend with `SymmetrixBackend.from_model(model_file, dtype="float32")` instead of passing a calculator.

Device memory of Symmetrix calculators (MACE-MP-0b medium, FP32, RTX 4090), measured with `nvidia-smi`:

| Calculators and structure | Device memory |
| --- | --- |
| One calculator, nothing evaluated (weights and runtime) | 643 MiB |
| One calculator after evaluating 1000 Si atoms | 919 MiB |
| One calculator after evaluating 10,648 Si atoms | 1391 MiB |
| Two calculators, each after 10,648 atoms | 2187 MiB |
| Four calculators, each after 10,648 atoms | 3783 MiB |

Each further calculator costs its own copy of the weights and its own workspace, which grows by about 50 KB per atom: about 0.8 GB for 10,648 atoms. With the default `max_calculators=4` a run therefore needs at most about 2.4 GB more than with the shared calculator alone; on smaller GPUs or with many concurrent large trajectories, lower `max_calculators` (0 keeps every structure on the shared calculator). The shared calculator's workspace grows with the total number of atoms in a joined call, so batching many structures also costs memory.

[TorchSim](https://github.com/TorchSim/torch-sim) avoids per-trajectory state altogether: it rebuilds the neighbor lists of all systems on the GPU at every call and keeps positions on the GPU, because its integrators run there too. Symmetrix-XL 0.1.1 can only hold the device graph of one structure per calculator, which is why atombatch uses dedicated calculators for large trajectories.

## What a round costs

The scheduler evaluates once every running task is either waiting on a request or finished. A round therefore costs the batched model calls plus the Python step logic of all running tasks (optimizer updates, NEB tangents, integrator steps). That step logic runs under the GIL, one task after another. The model call usually dominates.

With a model that costs almost nothing (a harmonic potential), a FIRE step on a 40-atom cell cost about 210 to 320 us in a plain ASE loop on the M1 Max, and about 100 us per evaluation through atombatch, independent of `max_active` (1, 8, or 64). Thread hand-offs and the scheduler itself therefore add almost nothing. Most of ASE's cost was `Calculator.check_state`, which ASE calls about four times per step and which compares every array of the structure with `np.allclose`; for small arrays each call costs about 15 us of Python and NumPy overhead. Since 0.1.2, `BatchCalculator.check_state` first compares arrays for exact equality and falls back to ASE's tolerance only when they differ, which gives the same result as ASE and cut atombatch's time per evaluation from about 260 to 100 us. The remainder is mostly copying `Atoms` and the optimizer update.

Inside a batched Symmetrix call, building neighbor lists on the CPU took about 40 % of the call for 64 cells of 40 atoms on the M1 Max (46 of 112 ms), because every structure was given a fresh list at every step. The Symmetrix ASE calculator avoids this with a neighbor skin (0.5 A by default): it builds the list with `cutoff + neighbor_skin` and rebuilds only when atoms have moved more than half the skin. Since 0.1.2, atombatch keeps one such Symmetrix neighbor cache per trajectory, so each task's edges are those the Symmetrix calculator would use for the same trajectory, and the 64-cell call took 79 instead of 147 ms in a relaxation-like sequence. The throughput table above feeds unrelated structures and is not affected.

With the cache in place, a 64-relaxation run built neighbor lists only once per trajectory (0.04 s in total on the M1 Max), so a faster neighbor-list library would gain almost nothing. The remaining cost was updating the cached edges at every step: Symmetrix recomputes the edge vectors and lengths of about 3800 edges per 40-atom cell in several NumPy passes, about 200 us per structure. atombatch lets Symmetrix decide whether to reuse or rebuild the cache and computes the edge geometry itself in one loop compiled with Numba (`atombatch.backends._edges`), which takes 74 us per structure. The loop performs the same floating-point operations in the same order as Symmetrix, without fused multiply-adds, so its edges are identical bit for bit; the tests check this over reused, rebuilt, and cell-reset caches. Without Numba, a NumPy version with the same results is used. This reduced the batched relaxations from 4.3 to 3.5 s on the M1 Max and from 2.0 to 1.8 s on the RTX 4090 (FP32). Finally, the graph is passed to Symmetrix as contiguous `int32` arrays, which its bindings accept without copying; Python lists of the roughly 250,000 neighbor types per call had been converted element by element. On the RTX 4090 this reduced the FP32 relaxations from 1.8 to 1.6 s.

{attr}`~atombatch.Scheduler.batch_log` records every call as `(model, structures, atoms)` and is the first thing to check: it shows whether batches are as large as intended.

## Settings

`max_active`
: The number of tasks running at once. Larger values give larger batches but more serialized Python time per round. On the M1 Max GPU, throughput was best at about 64 structures of 40 atoms per call and fell at 128. On the CPU, MACE was fastest at 16 to 32. The best value depends on the model, the device, and the structure size, so measure it with `benchmarks/bench_batching.py`. Set it explicitly for large runs, because `None` starts one thread per submitted task.

`max_atoms`
: The atom budget of one call, set per model in {meth}`~atombatch.Scheduler.add_model`. Choose it from GPU memory, and ideally close to `max_active` times the typical structure size, so that most rounds make one call.

## Things that split or shrink batches

Stress
: Requests that need stress and requests that do not are evaluated in separate calls. Relaxing with and without a cell filter in the same run costs two calls per model per round.

The tail of a run
: Finished tasks are replaced from the queue only while it has tasks left. Near the end, the batch shrinks to the tasks still running. Tasks are admitted in submission order, so submitting the longest tasks first shortens the tail.

Slow code inside tasks
: A round waits for every running task. MD observers that write files, or analysis inside a task, delay all other tasks. Keep observer intervals sparse and move analysis after {meth}`~atombatch.Scheduler.run`.

Sequential evaluation inside a workflow
: A workflow that evaluates independent structures one at a time contributes one structure per round. Use `context.evaluate` for those evaluations (see {doc}`workflows`).

## Large cells

A cell large enough to fill the GPU gains little from batching, because its per-call overhead is already amortized over many atoms. Splitting such a cell into pieces does not help either: each piece needs a halo of ghost atoms at least as wide as the model's receptive field (for MACE-MP, the 6 Å cutoff times two message-passing layers), which multiplies the work for pieces that are not much larger than the halo. For cells that exceed one GPU's memory, use LAMMPS with the Symmetrix or MACE interface, which implements domain decomposition with halo exchange.
