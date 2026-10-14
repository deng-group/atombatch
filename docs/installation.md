# Installation

atombatch requires Python 3.10 or later, with NumPy and ASE 3.23 or later as its only runtime dependencies. It is not yet on PyPI, so install it from a clone of the repository:

```bash
git clone https://github.com/deng-group/atombatch.git
cd atombatch
uv venv && source .venv/bin/activate
uv pip install -e .
```

Optional extras add the packages for batched models and workflows:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `symmetrix` | `symmetrix-xl` (0.1.1 or later) | batched evaluation with Symmetrix |
| `torchsim` | `torch-sim-atomistic` (Python 3.12 or later) | batched MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models |
| `phonons` | `phonopy`, `phono3py` | {func}`~atombatch.displacement_forces` |
| `test` | `pytest` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-parser` | building this documentation |

For example, `uv pip install -e ".[torchsim,phonons]"`. A TorchSim model also needs its model package, such as `mace-torch` for MACE.

## Running the tests

```bash
uv pip install -e ".[test]"
pytest
```

The tests use ASE's EMT calculator and need no MLIP. Tests for TorchSim, phonopy, and phono3py are skipped when those packages are absent. Two tests run only on request: `ATOMBATCH_MACE_MODEL=small` compares batched MACE with MACE's own ASE calculator, and `ATOMBATCH_SYMMETRIX_MODEL=/path/to/model.json` does the same for Symmetrix.

## Building the documentation

```bash
uv pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
```
