# Re-analysis pipeline

Everything in this directory was written for the JSS revision. It supersedes
`results/scripts/`, which contains the original plotting code (kept for
provenance, marked deprecated: it reads CodeCarbon's kWh column and labels it
Joules).

## Running it

```bash
pip install pandas numpy scipy matplotlib seaborn tabulate
cd results/analysis
./run_all.sh
```

Output goes to `results/revision/tables/` and `results/revision/figures/`.

## What each script does

| script | what it settles |
|---|---|
| `common.py` | Single loader. Converts CodeCarbon kWh to Joules once, names ecosystems consistently, defines the GPU-only and harmonised energy boundaries. |
| `01_data_audit.py` | Proves the native unit from the data; the real run/epoch counts; GPU load; instrument inconsistencies across ecosystems; implausible power samples; within-run dispersion. |
| `02_main_tables.py` | Corrected Tables 3 and 4 in Joules, with medians and dispersion, the CPU/GPU/RAM breakdown, and how the ranking moves with the energy definition. |
| `03_statistics.py` | Omnibus and pairwise tests with effect sizes, under an explicit pseudo-replication caveat; the shared-backend LibTorch control group. |
| `04_energy_vs_time.py` | RQ3 quantified: within-ecosystem correlation, cross-ecosystem rank agreement, and the power/duration decomposition. |
| `05_efficiency_index.py` | Alpha sweep, one consistent normalisation, and the Pareto set. |
| `06_dataset_scaling.py` | Absolute-scale dataset comparison and the factors "complexity" conflates. |
| `07_carbon_and_scale.py` | Recomputed campaign footprint; the fleet extrapolation with its energy boundary made explicit. |
| `08_figures.py` | Corrected figures: one unit, per-model/dataset panels, absolute dataset scale, Pareto frontier, GPU load. |
| `09_campaign_v2.py` | Between-run statistics and quality-normalised efficiency. Requires the replicated campaign (`scripts/run_campaign.py`). |

`repetition_protocol.md` specifies how the replicated campaign must be run.

## Headline corrections

1. `energy_consumed` is **kWh**, not Joules — a factor of 3.6e6. Verified three
   ways in `01_data_audit.py`.
2. The campaign contains **2,880** tracked blocks (1,440 training + 1,440
   inference), not 7,200, and **one** run per configuration.
3. The instrument was **not the same** across ecosystems: two CodeCarbon major
   versions, two tracking modes, two sampling intervals. For the CodeCarbon
   2.8.4 stacks roughly two thirds of the reported energy is a constant host
   power times duration.
4. The four LibTorch stacks span **98–100%** of the total reported spread, so the
   effect is host-side, not language-intrinsic.
5. Under a common measurement boundary the energy and time rankings agree almost
   perfectly (Spearman 1.00 for inference, 0.98 for training), which does not
   support "faster is not greener" as stated.
