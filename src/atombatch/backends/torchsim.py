"""Batched evaluation of TorchSim models (MACE, fairchem, SevenNet, ORB, ...)."""

import sys

from ase.stress import full_3x3_to_voigt_6_stress

from .base import Backend


class TorchSimBackend(Backend):
    """Evaluate structures in one call of a TorchSim ``ModelInterface``.

    TorchSim provides batched interfaces for MACE, fairchem, SevenNet, ORB,
    MatterSim, NequIP, Nequix, metatomic, and classical potentials. The
    structures are joined into one ``SimState`` on the model's device and
    dtype, and the outputs are split back per structure. Stress requires a
    model constructed with ``compute_stress=True``.
    """

    def __init__(self, model, max_atoms=None):
        self.model = model
        self.max_atoms = max_atoms

    def evaluate(self, structures, properties):
        import torch_sim

        if "stress" in properties and not self.model.compute_stress:
            raise ValueError(
                "Stress was requested from a TorchSim model constructed "
                "without compute_stress=True."
            )
        state = torch_sim.io.atoms_to_state(
            list(structures), device=self.model.device, dtype=self.model.dtype
        )
        output = self.model(state)
        energies = _numpy(output["energy"]).reshape(-1)
        forces = _numpy(output["forces"])
        stresses = _numpy(output["stress"]) if "stress" in properties else None
        results, start = [], 0
        for index, atoms in enumerate(structures):
            stop = start + len(atoms)
            result = {
                "energy": float(energies[index]),
                "forces": forces[start:stop].copy(),
            }
            if stresses is not None:
                result["stress"] = full_3x3_to_voigt_6_stress(stresses[index])
            results.append(result)
            start = stop
        return results


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _is_torch_sim_model(model):
    # Checked only when TorchSim is already imported, so that ASE-only runs
    # do not import torch.
    interface = sys.modules.get("torch_sim.models.interface")
    return interface is not None and isinstance(model, interface.ModelInterface)
