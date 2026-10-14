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
| Symmetrix FP32 | 62 | 15 | 6.5 | 5.3 | 18 | 11.7x |
| Symmetrix FP64 | 140 | 43 | 28 | 26 | 38 | 5.5x |
| MACE via TorchSim FP32 | 204 | 55 | 46 | 44 | 44 | 4.6x |
| MACE via TorchSim FP64 | 298 | 209 | 193 | 191 | out of memory | 1.6x |

Model throughput in us/atom against structures per call. Relaxations of 64 cells with `max_active=64`:

| Model and precision | Sequential | atombatch (model + other) | Speedup |
| --- | --- | --- | --- |
| Symmetrix FP32 | 6.5 s | 1.8 s (1.1 + 0.7) | 3.7x |
| Symmetrix FP64 | 14.3 s | 4.0 s (3.4 + 0.6) | 3.6x |
| MACE via TorchSim FP32 | 23.4 s | 5.8 s (5.1 + 0.7) | 4.1x |
| MACE via TorchSim FP64 | 34.2 s | 23.1 s (22.4 + 0.7) | 1.5x |

All runs took the same optimizer steps as the sequential loop. FP64 is slow on this consumer GPU, which leaves MACE compute-bound and gains little from batching. With atombatch 0.1.1, the time outside the model call was 1.4 to 2.0 s in these runs; with Symmetrix in FP32 it was 43 % of the run and limited the relaxation speedup to 1.5x. The faster `check_state` reduced it to about 0.65 s (0.23 ms per evaluation) and raised that speedup to 2.4x.

Three conclusions follow from these numbers.

- On the GPUs, batching reduced the cost per atom 5 to 12 times at 64 structures per call (FP32 Symmetrix and MACE, and FP64 Symmetrix) and made the relaxations 3.6 to 4.8 times faster. The relaxation gain is smaller than the throughput gain because relaxations finish at different times, so the mean batch (44) is below `max_active` and the last rounds are nearly empty. Throughput fell again at 128 structures per call, so larger is not always better.
- The time outside the model call is now 0.1 to 0.25 ms per evaluation, 6 to 22 % of the GPU runs. Most of it is ASE's own per-step work (see below), and on fast GPUs it still sets the limit for small cells.
- On the CPU, the gain depends on the model and on the batch size. The single-core Symmetrix build is compute-bound and gains nothing. MACE through TorchSim with 8 threads gained about 1.6 times at 16 to 32 structures per call and lost at 64 and above, so `max_active=16` gave a 1.8x speedup where `max_active=64` gave a slowdown. On a CPU, measure before choosing `max_active`.

## What a round costs

The scheduler evaluates once every running task is either waiting on a request or finished. A round therefore costs the batched model calls plus the Python step logic of all running tasks (optimizer updates, NEB tangents, integrator steps). That step logic runs under the GIL, one task after another. The model call usually dominates.

With a model that costs almost nothing (a harmonic potential), a FIRE step on a 40-atom cell cost about 210 to 320 us in a plain ASE loop on the M1 Max, and about 100 us per evaluation through atombatch, independent of `max_active` (1, 8, or 64). Thread hand-offs and the scheduler itself therefore add almost nothing. Most of ASE's cost was `Calculator.check_state`, which ASE calls about four times per step and which compares every array of the structure with `np.allclose`; for small arrays each call costs about 15 us of Python and NumPy overhead. Since 0.1.2, `BatchCalculator.check_state` first compares arrays for exact equality and falls back to ASE's tolerance only when they differ, which gives the same result as ASE and cut atombatch's time per evaluation from about 260 to 100 us. The remainder is mostly copying `Atoms` and the optimizer update.

Inside a batched Symmetrix call, building neighbor lists on the CPU took about 40 % of the call for 64 cells of 40 atoms on the M1 Max (46 of 112 ms), because every structure was given a fresh list at every step. The Symmetrix ASE calculator avoids this with a neighbor skin (0.5 A by default): it builds the list with `cutoff + neighbor_skin` and rebuilds only when atoms have moved more than half the skin. Since 0.1.2, atombatch keeps one such Symmetrix neighbor cache per trajectory, so each task's edges are those the Symmetrix calculator would use for the same trajectory, and the 64-cell call took 79 instead of 147 ms in a relaxation-like sequence. The throughput table above feeds unrelated structures and is not affected.

With the cache in place, a 64-relaxation run built neighbor lists only once per trajectory (0.04 s in total on the M1 Max), so a faster neighbor-list library would gain almost nothing. The remaining cost was updating the cached edges at every step: Symmetrix recomputes the edge vectors and lengths of about 3800 edges per 40-atom cell in several NumPy passes, about 200 us per structure. atombatch lets Symmetrix decide whether to reuse or rebuild the cache and computes the edge geometry itself in one loop compiled with Numba (`atombatch._edges`), which takes 74 us per structure. The loop performs the same floating-point operations in the same order as Symmetrix, without fused multiply-adds, so its edges are identical bit for bit; the tests check this over reused, rebuilt, and cell-reset caches. Without Numba, a NumPy version with the same results is used. This reduced the batched relaxations from 4.3 to 3.5 s on the M1 Max and from 2.0 to 1.8 s on the RTX 4090 (FP32).

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
