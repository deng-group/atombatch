# Design notes

These notes record the design decisions taken before implementation.

## Execution model

Each task runs its unmodified ASE code in its own worker thread, with a proxy calculator attached to every structure it evaluates. When the task requests energies or forces for a state that has no current result, the proxy submits a request and blocks. The scheduler waits until every running task is blocked or finished, evaluates all pending requests in one batched call per model, delivers the results, and resumes the tasks.

This handles algorithms that evaluate more than once per step: line-search optimizers, velocity-Verlet and Langevin integrators that request forces at the new positions inside `step()`, and NEB options that move images while computing forces. A lockstep design that evaluates only between steps, as in the Symmetrix `relax_batch` and `neb_batch` prototypes and MatterSim's `BatchRelaxer`, rejects those cases.

Python step logic runs under the GIL, but it is short compared with the batched evaluation. Its per-structure cost (a fraction of a millisecond per step) sets a floor that matters for structures of a few tens of atoms on a GPU and must be measured.

## Scheduling

- A per-model budget in atoms or edges bounds each batched call; requests beyond the budget wait for the next round.
- Finished tasks are replaced from the submission queue, so batches stay full instead of shrinking as relaxations converge.
- Results are identical to independent runs, because every request is evaluated exactly at the state the task asked for.

## Backends

A backend maps a list of ASE `Atoms` to a list of results (`energy`, `forces`, and optionally `stress`, `energies`). Requests are grouped by model, because one batched call evaluates one set of weights.

- Symmetrix: `Symmetrix.calculate_batch`, one disconnected graph per call with per-structure stresses reduced natively. Requires a Symmetrix release that includes batched evaluation.
- Generic ASE calculator: sequential evaluation behind the same interface. It provides scheduling and refill but no batching speedup.
- Further MLIP backends are added one at a time, each validated against that model's own ASE calculator.

## Validation

- Every task type is compared with an independent ASE run of the same algorithm: identical step counts and positions to rounding in FP64.
- Mixed batches (relaxation, NEB, MD, single points in one run), refill, budgets, and error propagation from a failing task are tested with ASE's EMT calculator, which needs no MLIP.
- Performance is reported in us/atom against batch size, with atom counts, model cutoff, and edge counts.
