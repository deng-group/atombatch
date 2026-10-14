# Benchmarks

`bench_batching.py` compares batched with sequential model evaluation on rattled SrTiO3 supercells (40 atoms by default). It reports two measurements:

- Model throughput: us/atom of one batched call against the same structures evaluated one at a time through the model's ASE calculator, for batch sizes from 1 to 128.
- Relaxations: 64 FIRE relaxations run as a plain sequential ASE loop and through atombatch with `max_active=64`. The atombatch wall time is split into time in the batched model call and everything else (optimizer step logic, scheduler, thread hand-offs). The script also checks that both runs take the same number of steps.

```bash
python benchmarks/bench_batching.py --model symmetrix --device metal --path model.json --output benchmarks/results/name.json
python benchmarks/bench_batching.py --model symmetrix --device cpu --path model.json
python benchmarks/bench_batching.py --model mace --device cpu --path small
```

Symmetrix needs a build with `calculate_batch`, and `--device metal` needs the Apple GPU backend. The Symmetrix model must cover Sr, Ti, and O. `--model mace` uses MACE-MP through TorchSim and needs `torch-sim-atomistic` and `mace-torch`.

`results/` holds the JSON output of the runs quoted in the documentation (`docs/performance.md`).
