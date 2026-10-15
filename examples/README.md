# Examples

Each script runs one batched workflow with Symmetrix and prints a short summary. They expect a Symmetrix JSON model in `SYMMETRIX_MODEL`; the universal MACE-MP-0b medium model covers every element used here:

```bash
curl -L -o mace_agnesi_medium.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model
symmetrix_extract_mace --model mace_agnesi_medium.model --output mace-mp-0b-medium.json
export SYMMETRIX_MODEL=$PWD/mace-mp-0b-medium.json
python relax_orderings.py
```

On macOS the scripts use the Metal backend (FP32); with a Symmetrix CUDA package they run on the GPU automatically.

| Script | Workflow | Batched per call | Checks in the output | M1 Max | RTX 4090 |
| --- | --- | --- | --- | --- | --- |
| `relax_orderings.py` | Cell relaxation of all 28 Ba orderings in a 2x2x2 SrTiO3 supercell | all orderings still relaxing | symmetry-equivalent orderings reach the same energy (groups of 12, 12, 4) | 5 s | 6 s |
| `neb_vacancy.py` | Climbing-image NEB of an oxygen-vacancy hop in 3x3x3 SrTiO3, pure and next to Ba | endpoints, then all images of three bands (up to 21) | the two Ba hops are mirror images: reaction energies +0.132 and -0.132 eV | 8 s | 3 s |
| `md_temperatures.py` | Langevin MD of SrTiO3 with an oxygen vacancy at four temperatures, with observers | one structure per temperature | oxygen displacements grow with temperature | 50 to 80 s | |
| `phonons.py` | Phonopy force sets of cubic SrTiO3 at five lattice constants | all 15 displaced supercells in one call | the R-point instability grows on compression and the polar Gamma mode softens on expansion | 1 s | |
| `atomate2_relax.py` | The 28 orderings of `relax_orderings.py` as atomate2 `ForceFieldRelaxMaker` jobs, one after another and batched | all jobs still relaxing | same steps and documents; 2.4x faster batched | 25 s and 10 s | |
| `swap_monte_carlo.py` | Sr/Ba swap Monte Carlo with replica exchange at four temperatures in 3x3x3 (Sr,Ba)TiO3 | the trial configurations of all temperatures | swap acceptance rises with temperature | 16 to 28 s | |

Times include starting the model, and on CUDA the one-time compilation of its kernels. The physical results are illustrations of the workflows with an untuned universal model, not converged studies: the cells are small, Monte Carlo samples a fixed lattice without relaxation, and the MD runs are too short for diffusion coefficients.

`run_gpu.slurm` is a template for running a script on one GPU node of a SLURM cluster.
