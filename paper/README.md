# Manuscript

`main.tex` plus `sections/`. Every number in the text comes from the analysis
pipeline and can be regenerated:

```bash
./results/analysis/run_all.sh          # tables and figures -> results/revision/
python3 scripts/check_consistency.py   # the 56 conformance checks
```

## Where each claim comes from

| Claim in the paper | Produced by |
|---|---|
| Unit audit, 2,880 blocks, 4.02 kWh / 1.33 kg | `01_data_audit.py` |
| Instrument table, host shares, 231 W constant | `01_data_audit.py`, `audit_measurement_boundary.md` |
| GPU power 83–197 W against a 350 W limit | `01_data_audit.py`, `audit_gpu_load.md` |
| Spreads under three energy definitions | `02_main_tables.py`, `table_headline_spreads.md` |
| Spearman ρ and discordant pairs | `04_energy_vs_time.py`, `rq3_rank_inversions.md` |
| LibTorch control group, 98–100% | `03_statistics.py`, `stats_libtorch_control.md` |
| Protocol divergence table, loader ρ = −0.73 | `10_implementation_audit.py` |
| Defect catalogue | `impl_structural_findings.md` |
| Cross-stack accuracy table | one-epoch runs under the shared contract |

Numbers measured on the replication machine (RTX 3090) rather than by the
pipeline — the Rust synchronisation figures, the TensorFlow training-mode
accuracies, the cross-stack convergence table — are noted as such in the text.

## Building

```bash
cd paper && pdflatex main && pdflatex main
```

## What is deliberately not here

A corrected ranking of the ecosystems. The infrastructure is in place and all
seven stacks are verified on the accelerator, but the replicated campaign has not
been executed; see the threats section. A ranking produced before the defects are
fixed would inherit them.
