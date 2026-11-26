# KDA benchmarks

The repository keeps two reusable runners. Run from the repository root
with the local Python package and MUSA toolchain configured.

## Backward phase timing

```bash
PYTHONPATH=python python benchmark/bench_kda_bwd_phases.py
```

Times K1/K2/K3/K4 separately with MUSA events, excluding initial compilation
and warmup. The fixed BF16 case is B=1, T=2048, H=64, K=V=128. It reports
p50/p90 and an explicit K4 logical byte count.

