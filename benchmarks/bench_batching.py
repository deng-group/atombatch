"""Benchmark batched against sequential model evaluation.

Two measurements on rattled SrTiO3 supercells:

1. Model throughput: us/atom of one batched call against the same structures
   evaluated one at a time through the model's own ASE calculator.
2. Relaxations: FIRE relaxations of many cells, run as a plain sequential ASE
   loop and through atombatch. The atombatch wall time is split into time in
   the batched model call and everything else (Python step logic of the
   optimizers, scheduler, thread hand-offs).

Examples::

    python benchmarks/bench_batching.py --model symmetrix --device metal \
        --path ~/.cache/symmetrix/bench-models/<model>.json
    python benchmarks/bench_batching.py --model symmetrix --device cpu --path ...
    python benchmarks/bench_batching.py --model mace --device cpu --path small
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from atombatch import Backend, Scheduler, relax
from atombatch.backends import as_backend


def perovskite(repeat, seed):
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=np.eye(3) * 3.95,
        pbc=True,
    ).repeat(repeat)
    atoms.rattle(0.05, seed=seed)
    return atoms


class Timed(Backend):
    """Backend wrapper that accumulates the time spent in model calls."""

    def __init__(self, backend):
        self.backend = backend
        self.max_atoms = backend.max_atoms
        self.seconds = 0.0
        self.calls = 0

    def evaluate(self, structures, properties):
        start = time.perf_counter()
        results = self.backend.evaluate(structures, properties)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return results


def load_model(kind, device, path, dtype):
    """Return (batched model for add_model, factory of ASE calculators)."""
    if kind == "symmetrix":
        from symmetrix import Symmetrix

        options = {"dtype": "float32", "metal": True} if device == "metal" else {}
        options.setdefault("dtype", dtype)
        model = Symmetrix(path, **options)
        return model, lambda: Symmetrix(path, **options)
    if kind == "mace":
        import torch
        from mace.calculators.foundations_models import mace_mp
        from torch_sim.models.mace import MaceModel

        options = {"model": path, "default_dtype": dtype, "device": device}
        model = MaceModel(
            mace_mp(return_raw_model=True, **options),
            device=torch.device(device),
            dtype=getattr(torch, dtype),
            compute_stress=False,
        )
        calculator = mace_mp(**options)
        return model, lambda: calculator
    raise ValueError(kind)


def median_time(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def throughput(model, calculator_factory, repeat, batch_sizes, repeats):
    backend = as_backend(model)
    calculator = calculator_factory()
    structures = [perovskite(repeat, seed) for seed in range(max(batch_sizes))]
    atoms_per_cell = len(structures[0])

    def sequential(cells):
        for atoms in cells:
            atoms = atoms.copy()
            atoms.calc = calculator
            atoms.get_forces()

    sequential(structures[:2])
    backend.evaluate(structures[:2], ("energy", "forces"))
    single = median_time(lambda: sequential(structures[:16]), repeats) / 16
    rows = []
    for size in batch_sizes:
        cells = structures[:size]
        backend.evaluate(cells, ("energy", "forces"))
        batched = median_time(
            lambda cells=cells: backend.evaluate(cells, ("energy", "forces")), repeats
        )
        rows.append(
            {
                "atoms_per_cell": atoms_per_cell,
                "batch": size,
                "sequential_us_per_atom": 1e6 * single / atoms_per_cell,
                "batched_us_per_atom": 1e6 * batched / (size * atoms_per_cell),
            }
        )
    return rows


def relaxations(model, calculator_factory, repeat, count, max_active, fmax, steps):
    cells = [perovskite(repeat, 1000 + seed) for seed in range(count)]

    calculator = calculator_factory()
    start = time.perf_counter()
    sequential_steps = []
    for atoms in cells:
        atoms = atoms.copy()
        atoms.calc = calculator
        dynamics = FIRE(atoms, logfile=None)
        dynamics.run(fmax=fmax, steps=steps)
        sequential_steps.append(dynamics.nsteps)
    sequential_seconds = time.perf_counter() - start

    timed = Timed(as_backend(model))
    scheduler = Scheduler(max_active=max_active)
    scheduler.add_model("model", timed)
    for atoms in cells:
        scheduler.submit(relax(atoms, fmax=fmax, steps=steps, optimizer=FIRE), "model")
    start = time.perf_counter()
    results = scheduler.run()
    batched_seconds = time.perf_counter() - start
    assert all(result.ok for result in results), [r.error for r in results if not r.ok]
    evaluations = sum(size for _, size, _ in scheduler.batch_log)
    return {
        "cells": count,
        "atoms_per_cell": len(cells[0]),
        "max_active": max_active,
        "evaluations": evaluations,
        "batched_calls": len(scheduler.batch_log),
        "mean_batch": evaluations / len(scheduler.batch_log),
        "sequential_s": sequential_seconds,
        "atombatch_s": batched_seconds,
        "atombatch_model_s": timed.seconds,
        "atombatch_other_s": batched_seconds - timed.seconds,
        "speedup": sequential_seconds / batched_seconds,
        "same_steps": sequential_steps == [r.value["steps"] for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["symmetrix", "mace"], required=True)
    parser.add_argument("--device", default="cpu", help="cpu, metal, cuda, ...")
    parser.add_argument("--path", required=True, help="model file or MACE-MP name")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="model precision (Metal is always float32)",
    )
    parser.add_argument("--repeat", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--relax-cells", type=int, default=64)
    parser.add_argument("--max-active", type=int, default=64)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, calculator_factory = load_model(
        args.model,
        args.device,
        str(Path(args.path).expanduser()) if args.model == "symmetrix" else args.path,
        args.dtype,
    )
    rows = throughput(
        model, calculator_factory, args.repeat, args.batch_sizes, args.repeats
    )
    print(
        f"\nModel throughput ({args.model}, {args.device}, {rows[0]['atoms_per_cell']} atoms/cell)"
    )
    print(
        f"{'batch':>6} {'sequential us/atom':>19} {'batched us/atom':>16} {'speedup':>8}"
    )
    for row in rows:
        speedup = row["sequential_us_per_atom"] / row["batched_us_per_atom"]
        print(
            f"{row['batch']:>6} {row['sequential_us_per_atom']:>19.1f} "
            f"{row['batched_us_per_atom']:>16.1f} {speedup:>8.2f}"
        )

    relax_result = None
    if args.relax_cells:
        relax_result = relaxations(
            model,
            calculator_factory,
            args.repeat,
            args.relax_cells,
            args.max_active,
            args.fmax,
            args.steps,
        )
        r = relax_result
        print(
            f"\nFIRE relaxations: {r['cells']} cells x {r['atoms_per_cell']} atoms, max_active={r['max_active']}"
        )
        print(
            f"  evaluations {r['evaluations']} in {r['batched_calls']} batched calls (mean batch {r['mean_batch']:.1f})"
        )
        print(f"  sequential ASE loop  {r['sequential_s']:8.2f} s")
        print(
            f"  atombatch            {r['atombatch_s']:8.2f} s  (model {r['atombatch_model_s']:.2f} s, other {r['atombatch_other_s']:.2f} s)"
        )
        print(
            f"  speedup              {r['speedup']:8.2f}x   same step counts: {r['same_steps']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "device": args.device,
                    "dtype": "float32" if args.device == "metal" else args.dtype,
                    "path": Path(args.path).name,
                    "machine": platform.platform(),
                    "processor": platform.processor(),
                    "throughput": rows,
                    "relaxations": relax_result,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
