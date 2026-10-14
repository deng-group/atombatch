# Installation

atombatch requires Python 3.10 or later, with NumPy and ASE 3.23 or later as its only runtime dependencies. Install it from PyPI:

```bash
pip install atombatch
```

For development, install it from a clone of the repository:

```bash
git clone https://github.com/deng-group/atombatch.git
cd atombatch
uv venv && source .venv/bin/activate
uv pip install -e .
```

Optional extras add the packages for batched models and workflows:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `symmetrix` | `symmetrix-xl` (0.1.1 or later), `matscipy` | batched evaluation with Symmetrix |
| `torchsim` | `torch-sim-atomistic` (Python 3.12 or later) | batched MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models |
| `phonons` | `phonopy`, `phono3py` | {func}`~atombatch.displacement_forces` |
| `test` | `pytest` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-parser` | building this documentation |

For example, `pip install "atombatch[torchsim,phonons]"`. Symmetrix-XL 0.1.1 on PyPI has Linux wheels only. On macOS, install the `metal-preview-1` wheels from GitHub or build Symmetrix-XL from source, then install atombatch without the `symmetrix` extra and add `matscipy`. A TorchSim model also needs its model package, such as `mace-torch` for MACE.

## Symmetrix on NVIDIA GPUs

Symmetrix-XL publishes CUDA wheels per GPU architecture: `symmetrix-xl-cuda12-sm80` (A100), `-sm86` (RTX 30 series, A40), `-sm89` (RTX 40 series, L40), `-sm90` (H100), and `symmetrix-xl-cuda13-sm120` (RTX 50 series, B200). `symmetrix backend install` selects one automatically, and `symmetrix doctor` checks the installation.

```bash
pip install atombatch matscipy symmetrix-xl-cuda12-sm89
```

The 0.1.1 CUDA wheels load the CUDA runtime and NVRTC by their unversioned names, `libcudart.so` and `libnvrtc.so`. On a machine where an older CUDA toolkit is registered with the dynamic linker, these names resolve to that toolkit rather than to the CUDA 12 libraries installed with the wheel. Evaluation then fails with `Failed to load function: nvrtcGetCUBIN` (an NVRTC older than 11.1) or with `invalid device ordinal` from sphericart (a CUDA runtime that does not see the memory allocated by the CUDA 12 runtime). `symmetrix doctor` does not detect this. Point both names at the wheel's libraries:

```bash
site=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
mkdir -p ~/.local/lib/symmetrix-cuda
ln -sf $site/cuda_nvrtc/lib/libnvrtc.so.12 ~/.local/lib/symmetrix-cuda/libnvrtc.so
ln -sf $site/cuda_runtime/lib/libcudart.so.12 ~/.local/lib/symmetrix-cuda/libcudart.so
export LD_LIBRARY_PATH=~/.local/lib/symmetrix-cuda:$site/cuda_nvrtc/lib:$site/cuda_runtime/lib:$LD_LIBRARY_PATH
```

TorchSim on CUDA compiles small helpers with Triton at run time, which needs the Python development headers. A virtual environment created from a system Python without them fails with a `CalledProcessError` from `gcc`. Use a Python that ships its headers, for example one managed by uv (`uv venv --managed-python`).

## Running the tests

```bash
uv pip install -e ".[test]"
pytest
```

The tests use ASE's EMT calculator and need no MLIP. Tests for TorchSim, phonopy, and phono3py are skipped when those packages are absent. Two tests run only on request: `ATOMBATCH_MACE_MODEL=small` compares batched MACE with MACE's own ASE calculator (on the device in `ATOMBATCH_MACE_DEVICE`, default `cpu`), and `ATOMBATCH_SYMMETRIX_MODEL=/path/to/model.json` does the same for Symmetrix.

## Building the documentation

```bash
uv pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
```
