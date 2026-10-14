# Design notes

These notes record the design decisions taken before implementation.

## Execution model

Each task runs its unmodified ASE code in its own worker thread, with a proxy calculator attached to every structure it evaluates. When the task requests energies or forces for a state that has no current result, the proxy submits a request and blocks. The scheduler waits until every running task is blocked or finished, evaluates all pending requests in one batched call per model, delivers the results, and resumes the tasks.

This handles algorithms that evaluate more than once per step: line-search optimizers, velocity-Verlet and Langevin integrators that request forces at the new positions inside `step()`, and NEB options that move images while computing forces. A lockstep design that evaluates only between steps, as in the Symmetrix `relax_batch` and `neb_batch` prototypes and MatterSim's `BatchRelaxer`, rejects those cases.

Python step logic runs under the GIL, but it is short compared with the batched evaluation. Its per-structure cost (a fraction of a millisecond per step) sets a floor that matters for structures of a few tens of atoms on a GPU and must be measured.

## Batched workflows

A task blocks on one request at a time, so a workflow that evaluates its structures one after another contributes one structure per round. Workflows whose evaluations are independent of each other (displaced supercells for force constants, scaled cells for an equation of state, trial configurations of Monte Carlo replicas) instead submit them together with `TaskContext.evaluate`, which places all structures in one request. A single workflow then fills a batch on its own, and several such workflows share rounds as before.

Swap Monte Carlo evaluates one trial swap per replica per step, so the batch size equals the number of temperatures. Proposals are uniform over unlike pairs of sites, which is symmetric at fixed composition, and the Metropolis and replica-exchange tests use energies only. Positions stay on the input lattice. Relaxed-lattice sampling would need one relaxation per trial and is not implemented.

## Scheduling

- A per-model budget in atoms or edges bounds each batched call; requests beyond the budget wait for the next round.
- Finished tasks are replaced from the submission queue, so batches stay full instead of shrinking as relaxations converge.
- Results are identical to independent runs, because every request is evaluated exactly at the state the task asked for.

## Backends

A backend maps a list of ASE `Atoms` to a list of results (`energy`, `forces`, and optionally `stress`, `energies`). Requests are grouped by model, because one batched call evaluates one set of weights.

- Symmetrix: one disconnected neighbor graph per call. Releases with `Symmetrix.calculate_batch` evaluate it themselves. For Symmetrix-XL 0.1.1, which has no `calculate_batch`, `atombatch.symmetrix` builds the joined graph and calls the evaluator directly, using only evaluator calls present in 0.1.1, and reduces per-structure stresses from the pair forces in NumPy. Both paths agree with the Symmetrix ASE calculator to rounding and run at the same speed. Models with dispersion corrections or electric-field coupling fall back to sequential evaluation.
- TorchSim: one `ModelInterface` call on a `SimState` joined from the structures, with energies, forces, and 3x3 stresses split back per structure and stresses converted to Voigt order. TorchSim maintains batched interfaces for MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic, so one adapter covers those packages. Its stress follows the ASE sign convention. The model must be constructed with `compute_stress=True` for tasks that read stress.
- Function: a user callable with the backend signature, for models whose batched inference TorchSim does not wrap.
- Generic ASE calculator: sequential evaluation behind the same interface. It provides scheduling and refill but no batching speedup. The ASE `Calculator` interface evaluates one `Atoms` per call, and periodic structures with different cells cannot be joined into one `Atoms`, so batching requires access below the calculator to the model's own batched input.
- Further native backends (bypassing TorchSim) are added one at a time, each validated against that model's own ASE calculator.

## Validation

- Every task type is compared with an independent ASE run of the same algorithm: identical step counts and positions to rounding in FP64.
- Mixed batches (relaxation, NEB, MD, single points in one run), refill, budgets, and error propagation from a failing task are tested with ASE's EMT calculator, which needs no MLIP.
- Workflows are compared with sequential evaluation of the same structures (force constants from phonopy and phono3py, EOS fits). Swap Monte Carlo is checked against exact Boltzmann populations of a lattice model small enough to enumerate.
- Performance is reported in us/atom against batch size, with atom counts, model cutoff, and edge counts.
