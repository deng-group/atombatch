# Performance

This page reports the first measurements and explains what determines the cost of a run, so that the settings can be chosen and measured for a given workload.

## Measurements

All runs used an Apple M1 Max (10 CPU cores, 32-core GPU) and rattled SrTiO{sub}`3` supercells of 40 atoms, with `benchmarks/bench_batching.py` (see `benchmarks/README.md`). The sequential baseline evaluates one structure per call through the model's own ASE calculator. The relaxation benchmark runs 64 FIRE relaxations (`fmax=0.05`), once as a plain ASE loop and once through atombatch. Every batched run took the same number of optimizer steps as the sequential run.

Model throughput, in us/atom:

| Structures per call | 1 | 4 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| Symmetrix, Metal GPU (FP32), sequential 145 | 141 | 46 | 28 | 25 | 23 | 43 |
| Symmetrix, CPU (FP64, one core), sequential 1510 | 1500 | 1730 | 1110 | 1660 | 1280 | 1220 |
| MACE-MP small via TorchSim, CPU (FP64, 8 threads), sequential 1600 | 1770 | 1460 | 1000 | 1040 | 2950 | 4820 |

Symmetrix ran MACE-OMat-0 medium, built from a Symmetrix-XL source checkout after 0.1.1 with the Apple GPU backend. atombatch's graph path for 0.1.1 gave the same throughput on that build (23.7 against 23.1 us/atom at 64 structures). With the `metal-preview-1` Symmetrix wheel, which has Metal but no `calculate_batch` and therefore uses atombatch's graph path, throughput was 40 us/atom at 64 structures per call and the 64 relaxations ran 2.0 times faster than the sequential loop (10.6 against 20.9 s, identical step counts). The CPU rows vary by about 20 % between repeated runs, so differences of that size are noise.

Relaxations of 64 cells:

| Model and device | `max_active` | Mean batch | Sequential | atombatch (model + other) | Speedup |
| --- | --- | --- | --- | --- | --- |
| Symmetrix, Metal GPU | 64 | 44 | 17.2 s | 5.7 s (4.95 + 0.74) | 3.0x |
| Symmetrix, CPU | 64 | 44 | 162 s | 178 s (176.7 + 1.1) | 0.9x |
| MACE via TorchSim, CPU | 64 | 47 | 185 s | 268 s (266.5 + 1.5) | 0.7x |
| MACE via TorchSim, CPU | 16 | 15 | 226 s | 125 s (123.9 + 1.0) | 1.8x |

On an NVIDIA RTX 4090 (24 GB, Intel Core i9-13900KF host), with the Symmetrix-XL 0.1.1 CUDA wheel (`symmetrix-xl-cuda12-sm89`, batched through atombatch's graph path) and MACE-MP small through TorchSim 0.6.2 with torch 2.14 (CUDA 13.0):

| Model and precision | Sequential | 4 | 16 | 64 | 128 | Gain at 64 |
| --- | --- | --- | --- | --- | --- | --- |
| Symmetrix FP32 | 62 | 15 | 7.0 | 5.3 | 18 | 11.7x |
| Symmetrix FP64 | 131 | 42 | 28 | 25 | 38 | 5.1x |
| MACE via TorchSim FP32 | 206 | 55 | 45 | 44 | 44 | 4.7x |
| MACE via TorchSim FP64 | 298 | 209 | 193 | 192 | out of memory | 1.6x |

Model throughput in us/atom against structures per call. Relaxations of 64 cells with `max_active=64`:

| Model and precision | Sequential | atombatch (model + other) | Speedup |
| --- | --- | --- | --- |
| Symmetrix FP32 | 6.6 s | 4.5 s (2.5 + 2.0) | 1.5x |
| Symmetrix FP64 | 14.3 s | 6.3 s (4.6 + 1.7) | 2.3x |
| MACE via TorchSim FP32 | 23.8 s | 6.7 s (5.3 + 1.4) | 3.5x |
| MACE via TorchSim FP64 | 34.1 s | 24.1 s (22.6 + 1.5) | 1.4x |

All runs took the same optimizer steps as the sequential loop. With Symmetrix in FP32, the batched model call became so fast that the time outside it (1.95 s, or 0.7 ms per evaluation on this host) was 43 % of the run, which limits the relaxation speedup to 1.5x despite an 11.7x gain in model throughput. FP64 is slow on this consumer GPU, which leaves MACE compute-bound and gains little from batching.

Three conclusions follow from these numbers.

- On the GPU, batching reduced the cost per atom about six-fold at 64 structures per call and made the relaxations three times faster. The relaxation gain is smaller than the throughput gain because relaxations finish at different times, so the mean batch (44) is below `max_active` and the last rounds are nearly empty. Throughput fell again at 128 structures per call, so larger is not always better.
- The time outside the model call (optimizer steps, scheduler, thread hand-offs) was 0.3 to 0.7 ms per evaluation. With the M1 Max GPU it was 13 % of the run. With Symmetrix in FP32 on the RTX 4090 it reached 43 %, so on fast GPUs this overhead, not the model, limits the gain for 40-atom cells.
- On the CPU, the gain depends on the model and on the batch size. The single-core Symmetrix build is compute-bound and gains nothing. MACE through TorchSim with 8 threads gained about 1.6 times at 16 to 32 structures per call and lost at 64 and above, so `max_active=16` gave a 1.8x speedup where `max_active=64` gave a slowdown. On a CPU, measure before choosing `max_active`.

## What a round costs

The scheduler evaluates once every running task is either waiting on a request or finished. A round therefore costs the batched model calls plus the Python step logic of all running tasks (optimizer updates, NEB tangents, integrator steps). That step logic runs under the GIL, one task after another. The model call usually dominates. In the measurements above, the time outside the model call was 0.3 to 0.5 ms per evaluation, which matters only for structures much smaller than 40 atoms on a fast GPU.

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
