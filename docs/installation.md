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
| `symmetrix` | `symmetrix-xl` (0.1.1 or later, from PyPI), `matscipy`, `numba` | batched evaluation with Symmetrix |
| `torchsim` | `torch-sim-atomistic` (Python 3.12 or later) | batched MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models |
| `phonons` | `phonopy`, `phono3py` | {func}`~atombatch.displacement_forces` |
| `test` | `pytest` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-parser` | building this documentation |

For example, `pip install "atombatch[torchsim,phonons]"`. A TorchSim model also needs its model package, such as `mace-torch` for MACE.

## Which Symmetrix-XL to install

The `symmetrix` extra installs the latest Symmetrix-XL release on PyPI (0.1.2), which has Linux wheels only. With it, atombatch batches standard MACE and MACE-MH-1 through its own joined graph. MACE-POLAR-1, the device-side joined batches of Symmetrix-XL 0.2, and the Apple GPU (Metal) backend need a source build. PyPI packages cannot depend on a Git branch, so install Symmetrix-XL separately:

| Symmetrix-XL | Source | Adds |
| --- | --- | --- |
| 0.1.2 | PyPI (`atombatch[symmetrix]`) | standard MACE and MACE-MH-1 |
| 0.2 | `v0.2.0-prepare` branch of `bonan-group/symmetrix-xl` | MACE-POLAR-1 (one structure at a time), joined batches evaluated on the device, Metal |
| 0.2 with batched MACE-POLAR-1 | `v0.2-deng` branch of `deng-group/symmetrix-xl` | `Symmetrix.calculate_polar_batch`, D3 in Polar batches, MACE-POLAR-1 tensor products on Metal |

atombatch detects what the installed Symmetrix-XL provides and uses the fastest path available; results are the same on every path.

### Apple silicon

The build takes a few minutes and needs the Xcode command line tools. Metal is enabled by default on macOS:

```bash
uv venv && source .venv/bin/activate
uv pip install atombatch matscipy numba \
  "symmetrix-xl @ git+https://github.com/deng-group/symmetrix-xl.git@v0.2-deng#subdirectory=symmetrix"
```

Pass `metal=True` to `symmetrix.Symmetrix` to evaluate on the GPU (FP32 only).

### NVIDIA GPUs

A plain `pip install` from source builds the CPU backend only. Build the CUDA backend with the Symmetrix-XL build frontend in a fresh environment, with `--arch` set to the compute capability of the GPU (`sm80` A100, `sm86` RTX 30 series, `sm89` RTX 40 series and L40, `sm90` H100, `sm120` RTX 50 series):

```bash
git clone --recursive -b v0.2-deng https://github.com/deng-group/symmetrix-xl.git
cd symmetrix-xl
uv venv && source .venv/bin/activate
python tools/symmetrix_build.py install --backend cpu --cpu-target native \
  --build-root build-cpu
python tools/symmetrix_build.py install --backend cuda --arch sm89 \
  --cuda-root /path/to/cuda --build-root build-cuda --generator "Unix Makefiles"
symmetrix doctor --json
uv pip install atombatch matscipy numba
```

Build on a node with the same GPU architecture and CPU type as the compute nodes, or set `--cpu-target x86-64-v3` for a portable CPU build. On clusters whose compute nodes have no internet access, run the build on a login node, or fill the uv cache there first and build with `UV_OFFLINE=1`. Symmetrix-XL's own `symmetrix/README.md` documents further options.

### Prebuilt CUDA wheels (0.1)

Symmetrix-XL 0.1 publishes CUDA wheels per GPU architecture: `symmetrix-xl-cuda12-sm80` (A100), `-sm86` (RTX 30 series, A40), `-sm89` (RTX 40 series, L40), `-sm90` (H100), and `symmetrix-xl-cuda13-sm120` (RTX 50 series, B200). `symmetrix backend install` selects one automatically, and `symmetrix doctor` checks the installation.

```bash
pip install atombatch matscipy numba symmetrix-xl-cuda12-sm89
```

The 0.1.1 CUDA wheels load the CUDA runtime and NVRTC by their unversioned names, `libcudart.so` and `libnvrtc.so`. On a machine where an older CUDA toolkit is registered with the dynamic linker, these names resolve to that toolkit rather than to the CUDA 12 libraries installed with the wheel. Evaluation then fails with `Failed to load function: nvrtcGetCUBIN` (an NVRTC older than 11.1) or with `invalid device ordinal` from sphericart (a CUDA runtime that does not see the memory allocated by the CUDA 12 runtime). `symmetrix doctor` does not detect this. Point both names at the wheel's libraries:

```bash
site=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
mkdir -p ~/.local/lib/symmetrix-cuda
ln -sf $site/cuda_nvrtc/lib/libnvrtc.so.12 ~/.local/lib/symmetrix-cuda/libnvrtc.so
ln -sf $site/cuda_runtime/lib/libcudart.so.12 ~/.local/lib/symmetrix-cuda/libcudart.so
export LD_LIBRARY_PATH=~/.local/lib/symmetrix-cuda:$site/cuda_nvrtc/lib:$site/cuda_runtime/lib:$LD_LIBRARY_PATH
```

## TorchSim on CUDA

TorchSim on CUDA compiles small helpers with Triton at run time, which needs the Python development headers. A virtual environment created from a system Python without them fails with a `CalledProcessError` from `gcc`. Use a Python that ships its headers, for example one managed by uv (`uv venv --managed-python`).

## Running the tests

```bash
uv pip install -e ".[test]"
pytest
```

The tests use ASE's EMT calculator and need no MLIP. Tests for TorchSim, phonopy, and phono3py are skipped when those packages are absent. Model tests run only on request: `ATOMBATCH_MACE_MODEL=small` compares batched MACE with MACE's own ASE calculator (on the device in `ATOMBATCH_MACE_DEVICE`, default `cpu`), and `ATOMBATCH_SYMMETRIX_MODEL=/path/to/model.json` does the same for Symmetrix, and `ATOMBATCH_SYMMETRIX_POLAR_MODEL=/path/to/MACE-POLAR-1.json` for MACE-POLAR-1.

## Building the documentation

```bash
uv pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
```
