# Analysis pipeline

Everything in this directory was written for the JSS revision. It supersedes
`results/scripts/`, which still holds the original plotting code (kept for
provenance; every file there carries a `DEPRECATED` header saying why — it
reads CodeCarbon's `energy_consumed` column, which is kWh, and labels it
Joules).

The pipeline has two halves. Scripts `01`–`08` and `10` audit the campaign of
the first submission, whose consolidated CodeCarbon log is
`results/data/combined_data.csv`. Scripts `09` and `11`–`19` analyse the
re-executed campaign in `results/campaign_v2/` and produce everything the
manuscript quotes.

## Running it

```bash
./results/analysis/run_all.sh     # the whole pipeline, in dependency order
./paper/build.sh                  # the pipeline, then tectonic
./paper/build.sh --no-data        # compile only, reusing paper/generated/
```

**Use the campaign interpreter, not whatever `python3` resolves to.**
`run_all.sh` resolves `.venv-deepgreen/bin/python` at the repository root and
exits 2 if it is not executable; set `PYTHON=/path/to/python` to override. The
system interpreter has no pandas, and under it `scripts/check_consistency.py`
reports a clean run on a repository it has not finished checking.
`paper/build.sh` passes the same `PYTHON` down and takes `TECTONIC` for the
LaTeX engine.

To run one script on its own, run it from this directory — each inserts its own
directory on `sys.path` to import `common`:

```bash
cd results/analysis && ../../.venv-deepgreen/bin/python 14_v2_statistics.py
```

Order matters in the second half. `11` writes the consolidated per-block table
that `14`, `16` and `13` read; `09` writes the quality table that `15` reads;
`12` and `13` read the output of all of them. Running them out of order
silently uses stale input.

## What it reads

| input | what it is |
|---|---|
| `results/data/combined_data.csv` | The first submission's campaign: 2,880 tracked blocks (1,440 training + 1,440 inference), eight ecosystems, one run per configuration. Read by `01`–`08` and `10`. |
| `results/campaign_v2/` | The re-executed campaign the manuscript reports: 210 run directories, 7 ecosystems × 2 models × 3 datasets × 5 repetitions, each holding `counters.csv` (NVML + RAPL, read by the harness at the phase boundaries), `emissions_*.csv` (CodeCarbon), `metrics.csv` and `manifest.json`. Not distributed — 13,000 files, gitignored; `scripts/consolidate_raw.py` flattens it into `results/replication/`. |
| `results/campaign_v2_first_campaign/` | The superseded replicated campaign, also 210 runs. An input to `18` and to `12`'s first-campaign macros, and nowhere else in the pipeline. |
| `results/calibration/` | Five runs of one configuration re-executed in a third window; the only input to `17`. `results/calibration_first_harness/` is the previous calibration, kept aside because it was produced under a different precision policy. |
| `results/gpu_utilisation.csv` | A 1 Hz `nvidia-smi` record; the input to `19`. It began after the campaign's first run, so it covers a subset, and `19` reports which. |
| `results/revision/record/` | Three CSVs of measurements no script here can regenerate: `initialiser_trials.csv`, `first_campaign_divergences.csv`, `vgg_fashion_pipeline_defect.csv`. `12` reads its `\vInit*`, `\vSeedDivergentStacks` and `\vPixelDivergentStacks` macros from them rather than having them typed into the manuscript, so each quoted number has exactly one place it can be corrected. |

## What it writes

| output | written by |
|---|---|
| `results/revision/tables/` | Every script, as a `.csv` and a `.md` side by side, through `common.save_table` (or `common.write_table_path` for the few tables built without it). |
| `results/revision/figures/` | `08` only. |
| `paper/generated/numbers.tex` | `12` — 328 `\newcommand` macros at this commit, one per value the manuscript quotes. |
| `paper/generated/tab_*.tex` | `12` — thirteen result tables (`tab_train_energy`, `tab_infer_energy`, `tab_accuracy`, `tab_energy_to_target`, `tab_control`, `tab_spread`, `tab_instrument`, `tab_coverage`, `tab_power_distortion`, `tab_mechanism`, `tab_padding`, `tab_industrial`, `tab_precision_contrast`). |
| `paper/figures/*.png` | `13` — six manuscript figures. |
| `results/replication/` | `scripts/consolidate_raw.py`, run at the *end* of `run_all.sh` because the replication package is an output of this pipeline, not an input to it. |

## The completeness gate

`common.py` holds one definition of what counts as a run, because there were
three and they disagreed.

* `EXPECTED_EPOCHS` — `DEEPGREEN_EPOCHS`, default 30. `read_complete_counters()`
  accepts a run directory only if its `counters.csv` records that many distinct
  epochs in *both* `train` and `eval`. A run that dies part-way leaves the
  blocks it managed behind; those are fragments, not small measurements of the
  same thing, and averaging them in produced ecosystem spreads of 20,000×.
* `EXPECTED_RUNS` — `DEEPGREEN_EXPECTED_RUNS`, default 210 (7 × 2 × 3 × 5).
  `campaign_status()` counts the complete directories under
  `results/campaign_v2/`; `campaign_is_partial()` compares the two.

## Partial campaigns divert

Monitoring a campaign still in flight is legitimate, and it once replaced the
manuscript's inputs with numbers from however many runs had finished — `\vRuns`
went from 210 to 59 in a single commit. So while the campaign is short of
`EXPECTED_RUNS`, every output goes somewhere else and the committed directories
are untouched:

| normal | while partial |
|---|---|
| `results/revision/tables/` | `results/revision/tables_partial/` |
| `results/revision/figures/` | `results/revision/figures_partial/` |
| `paper/generated/` | `paper/generated_partial/` |
| `paper/figures/` | `paper/figures_partial/` |

Reads are asymmetric. Each v2 script sets `TABLES = common.TABLES_RESOLVER`;
`TABLES / "x.csv"` resolves to the `_partial` sibling when this run has
produced that file there, and otherwise falls back to the committed directory.
A partial run therefore reads its own fresh output plus the stable
first-campaign tables, and overwrites neither. A script writing a table outside
`save_table` must go through `common.write_table_path`, which always writes to
`tables_dir()` — writing through the resolver would write into the committed
directory.

The fallback is right for a partial campaign and wrong when a script *dies*
before writing, so every script now writes every table it owns: a header and no
rows when there is nothing to report.

`common.announce_scope(name)` prints once per script how many runs are complete
and where that run may write.

## `--campaign v1`

Two scripts take it, both defaulting to `v2`:

```bash
./15_convergence.py --campaign v1     # -> v1_convergence_*.{csv,md}
./13_paper_figures.py --campaign v1   # -> paper/figures/fig_convergence_first_campaign.png
```

The training-collapse finding belongs to the superseded campaign — this one has
none — and the manuscript now presents it as that campaign's result, so both
have to be derivable. The v1 outputs are named apart from the v2 ones so
running either can never overwrite the other's tables, and `12` reads both and
reports the contrast. `run_all.sh` runs both invocations.

## `DEEPGREEN_MEASURE_HOST=1`

Three steps measure *this machine* rather than re-derive the campaign, so they
are host- and network-dependent and are listed separately in `run_all.sh`.
`17_window_calibration.py` runs unconditionally, because it only reads
directories that already exist. The two that actually re-measure the host —
`scripts/measure_idle.py` and `scripts/probe_reported_window.py` — run only
under `DEEPGREEN_MEASURE_HOST=1`, because this pipeline is itself load:
`measure_idle.py` used to run here unconditionally, and a full run measured the
idle host while being the only thing on it, writing 3.6 W of excess CPU package
power straight over a manuscript input.

Without the flag, `run_all.sh` prints when `v2_idle_baseline.json` and
`v2_window_mechanism.csv` were last measured, or warns that they are absent.
`12` reads both behind `if path.exists()`, so the `\vIdle*` and `\vMech*`
macros are simply not emitted when they have never been measured — which is why
`run_all.sh` says so out loud. Set the flag on a quiet host to re-measure.

`scripts/probe_tf32.py` is the other host measurement (the kernel-level
mechanism behind `18`). It writes `v2_tf32_ablation` through
`common.save_table` and is not in `run_all.sh`; `12` reads its table if it is
there.

## How the numbers reach the manuscript

`12_paper_numbers.py` exists so that no quoted value is typed by hand — the
honest answer for the submitted version was that the tables were transcribed
from spreadsheets, which is also how kilowatt-hours came to be labelled as
Joules. Two rules make it fail rather than mislead:

* **It refuses NaN.** `macro()` rejects any value whose string form is `nan`,
  `inf`, `none`, `<na>` and the rest. With zero collapses in the campaign,
  `chi2_contingency` raises on a zero expected frequency, `15_convergence`
  records NaN, and `paper.tex` had typeset "a chi-square test on that table
  returns p = nan, which would license a claim that ecosystems differ in
  robustness". The caller must give the quantity a defined value or not emit
  the macro at all and let LaTeX fail on the undefined command.
* **It counts PASS, and only PASS.** `apparatus_facts()` calls
  `check_consistency.run()` for the conformance figures. `\vConformancePassing`
  used to be `len(results) - len(failing)`, i.e. everything that did not fail,
  so a run in which three checks could not read their input would have
  published 92 of 92 passing. It now raises if any check reports `SKIP` — the
  conformance numbers may only come from a run in which every check actually
  ran — and counts `PASS` explicitly.

Several other guards are of the same kind: `only_row()` names the missing macro
instead of raising `IndexError`, and the first-campaign collapse macros raise a
readable error telling you to run `15_convergence.py --campaign v1` if that
campaign's tables are absent.

## What each script does

| script | reads | writes | what it settles |
|---|---|---|---|
| `common.py` | `results/data/combined_data.csv` on demand; walks `results/campaign_v2/` | nothing itself | The single point where kWh becomes Joules (`J_PER_KWH`), every column carrying its unit in its name. Also: ecosystem and backend naming, the completeness gate, `EXPECTED_RUNS` and the `_partial` diversion, the three energy definitions (as measured / GPU-only NVML / harmonised = GPU + 107 W uniform host model), `read_campaign_metrics`, and the statistics helpers — `t_ci` (used over the five runs of a configuration, where the percentile bootstrap is badly anti-conservative), `bootstrap_ci`, `cliffs_delta`, and `freeman_halton_exact`, an exact r×2 test of homogeneity for the collapse table, where chi-square's expected counts are far too small. |
| `01_data_audit.py` | `combined_data.csv` | `audit_units`, `audit_design`, `audit_gpu_load`, `audit_measurement_boundary`, `audit_outliers`, `audit_within_run_dispersion` | Proves the native unit of `energy_consumed` from the data itself (component additivity; emissions/energy recovering the Italian grid intensity only if the denominator is kWh). The real run, epoch and block counts. GPU load. The instrument configuration each ecosystem was actually measured under. Implausible sampled-power readings. Within-run dispersion. |
| `02_main_tables.py` | `combined_data.csv` | `table3_training_{as_measured,harmonised}`, `table4_inference_{as_measured,harmonised}`, `table_component_breakdown`, `table_headline_spreads`, `table_ranking_sensitivity_{training,inference}` | The submitted Tables 3 and 4 in Joules, with medians and dispersion beside means, the CPU/GPU/RAM breakdown, and how far the ranking moves when the energy definition changes. |
| `03_statistics.py` | `combined_data.csv` | `stats_omnibus`, `stats_pairwise_{training,inference}`, `stats_libtorch_control` | Omnibus and pairwise tests with effect sizes over the first campaign, under an explicit pseudo-replication caveat: each configuration ran once, so the 30 epochs are repeated measurements of one run and the effective n per configuration is 1. Plus the shared-backend LibTorch control group. |
| `04_energy_vs_time.py` | `combined_data.csv` | `rq3_within_ecosystem_correlation`, `rq3_rank_inversions`, `rq3_power_decomposition` | RQ3 quantified for the audited campaign: within-ecosystem correlation, cross-ecosystem rank agreement, and the decomposition into duration and mean power. Its tables carry eight ecosystems including MATLAB/DLT; the manuscript's RQ3 numbers come from `14`'s `v2_stats_energy_time`, not from here. |
| `05_efficiency_index.py` | `combined_data.csv` | `efficiency_index_alpha_sweep_{training,inference}`, `pareto_set_{training,inference}` | The Efficiency Index is arbitrary in α and inconsistently normalised (by the minimum in the tables, by the maximum in the heatmap), and its two components are near-collinear. Reports an α sweep, one consistent normalisation, and the Pareto set instead. |
| `06_dataset_scaling.py` | `combined_data.csv` | `dataset_factors`, `dataset_absolute_energy_{training,inference}`, `dataset_common_reference_{training,inference}` | Dataset scaling on an absolute scale, which per-dataset normalisation makes unreadable, and the four factors "dataset complexity" conflates. |
| `07_carbon_and_scale.py` | `combined_data.csv` | `campaign_carbon_footprint`, `fleet_boundary_sensitivity` | The campaign footprint recomputed over the true block count and stated as a lower bound, and the fleet extrapolation with its energy boundary made explicit. |
| `08_figures.py` | `combined_data.csv` | `results/revision/figures/fig_component_breakdown`, `fig_panels_{training,inference}`, `fig_pareto`, `fig_dataset_absolute`, `fig_gpu_load` | The first-campaign figures redrawn: one unit throughout, per-model/dataset panels, absolute dataset scale, a Pareto frontier in place of the qualitative quadrants, and the GPU/host energy split. Writes through `figures_dir()`, so it diverts like everything else. |
| `09_campaign_v2.py` | `results/campaign_v2/` run directories | `v2_between_run_statistics`, `v2_quality_normalised`, `v2_quality_summary` | Joins per-epoch CodeCarbon output to per-epoch quality metrics behind the completeness gate, making the independent run the unit of analysis. Between-run confidence intervals, accuracy per kilojoule, and energy to reach a target accuracy. `collect()` takes a campaign root so `18` can read the superseded campaign through the same gate rather than growing a second parser. |
| `10_implementation_audit.py` | `combined_data.csv`, plus the protocol read from the source of all eight stacks | `impl_protocol_divergence`, `impl_alignment_applied`, `impl_libtorch_versions`, `impl_parallelism_vs_ranking`, `impl_structural_findings` | Whether the comparison was like-for-like. It was not: the manuscript claims a shared learning rate, optimiser, batch size and epoch budget, and two of the eight stacks used a different learning rate and the loader parallelism ranged from 0 to 96 threads. Every cell cites the file and line it came from, and the second table records what the revision aligned. |
| `11_instrument_comparison.py` | every run directory's `counters.csv` and `emissions_*.csv` | `v2_instrument_epochs.csv` (the consolidated per-block table `14`, `16` and `13` read), `v2_instrument_summary`, `_by_ecosystem`, `_by_block`, `_ranking`, `_coverage`, `_windows`, `_duration_floor`, `_agreement_by_length`, `_power_distortion` | Every measured epoch carries two independent readings of the same interval — hardware counters bracketed at the phase boundaries, and CodeCarbon's own accounting. This compares them across the whole campaign rather than by spot check: where they agree (energy), where they disagree (the window), and where CodeCarbon is measuring something no counter on this machine can confirm (its modelled RAM term). |
| `12_paper_numbers.py` | `results/revision/tables/*` (41 tables), `results/revision/record/*`, both campaigns' metrics, `paper/paper.tex`, `scripts/check_consistency.run()` | `paper/generated/numbers.tex`, `paper/generated/tab_*.tex` | Emits every number the manuscript quotes as a LaTeX macro, so a number that changes in the data changes in the manuscript on the next build — or the build fails. See "How the numbers reach the manuscript" above for the NaN and PASS rules. |
| `13_paper_figures.py` | `v2_instrument_epochs`, `v2_between_run_statistics`, `v2_quality_normalised`, `v2_convergence_*` (and `v1_convergence_*` under `--campaign v1`) | `paper/figures/fig_window_floor`, `fig_energy_ci`, `fig_energy_accuracy`, `fig_instrument`, `fig_repeatability`, `fig_convergence_first_campaign` | The manuscript's figures. None of the four submitted ones survives as drawn: these show units, genuine between-run intervals, quality against energy, the two instruments as a function of phase length, and between-run coefficient of variation — the quantity the submitted single-run design could not estimate at all. |
| `14_v2_statistics.py` | `v2_instrument_epochs.csv` | `v2_run_totals.csv`, `v2_stats_omnibus`, `_pairwise`, `_libtorch_control`, `_energy_time`, `_cell_rho`, `_phase_consistency` | The inferential statistics both reviewers asked for, computed on run totals: five independent numbers per configuration instead of 30 autocorrelated epochs. Kruskal-Wallis within each (model, dataset, phase) block, pairwise Mann-Whitney with Holm correction and Cliff's delta, the shared-module control group (`SHARED_MODULE`, which excludes R because the R binding cannot load the exported module), and the energy–time relationship on counter-bracketed durations rather than the estimator's own duration field. |
| `15_convergence.py` | `results/campaign_v2/` or `results/campaign_v2_first_campaign/`, plus `<prefix>_quality_normalised` | `<prefix>_convergence_by_model`, `_by_ecosystem`, `_signature`, `_homogeneity`, `_conditional`, `_waste` (prefix `v2` or `v1`) | VGG-16 does not always train: in a fraction of runs it converges to exactly chance accuracy and stays there, having spent the full epoch budget. Whether it does is decided by the seed, which a single-run design cannot see. Two consequences: conditional on converging the ecosystems land within a couple of points of one another, so the apparent cross-stack accuracy spread is almost entirely collapsed runs; and fixed-budget energy comparison breaks, because a collapsed run costs full energy and produces nothing. The homogeneity test is `common.freeman_halton_exact`, not chi-square and not a permutation statistic. |
| `16_coverage_sensitivity.py` | `v2_instrument_epochs.csv` | `v2_coverage_per_block.csv`, `v2_coverage_by_ecosystem`, `_gap_attribution`, `_window_model`, `_window_diurnal`, `_window_rejected_fits`, `_power_distortion` | Tracked time ranges from about 40 % of wall time to almost 100 %, which reads as the low-coverage stacks doing a lot of untracked work. This establishes that they do not: the gap before each block matches the amount by which CodeCarbon's reported window exceeds the counter-bracketed phase, to a couple of hundred milliseconds and at r > 0.97 over every block. The untracked time is the instrument holding its window open, so coverage is a property of the apparatus and charging it to the ecosystems would charge them for the cost of being measured. |
| `17_window_calibration.py` | `results/calibration/`, `results/campaign_v2/` and `v2_instrument_epochs.csv` (11's output) | `v2_window_calibration` | Some runs were executed in a later window, so for those the interleaved design's protection against machine drift does not hold. This measures the drift instead of waving at it: one already-completed configuration re-executed in a third window, compared run for run against the within-window spread. `comparability_objection()` refuses the comparison outright when the two windows do not record the same precision policy, or when every calibration run predates the campaign's earliest run — a calibration older than the campaign is a comparison across harness versions, and it once read −0.21 % drift where the honest answer was that it was measuring a TF32 flag. Run start times come from `manifest.json`'s `machine_state.utc`, not from a file mtime that does not survive a copy. |
| `18_precision_ablation.py` | `results/campaign_v2/` and `results/campaign_v2_first_campaign/`, both through `09.collect()` | `v2_tf32_campaign_contrast` | The two campaigns already contain a precision ablation with a control group: Python/PyTorch ran with TF32 denied in the first and allowed in the second, while six stacks that never import that harness did not change, on the same configurations at 150 training epochs per cell. Most of the work is comparability — VGG-16 is out entirely (it ran as four different networks), TensorFlow and Java are out on architecture, and Fashion-MNIST and Tiny ImageNet are out because their *training* splits were re-encoded to 32×32 between the campaigns. Each cell is graded, and `12` uses only the comparable ones. |
| `19_gpu_utilisation.py` | `results/gpu_utilisation.csv`, plus each run's `manifest.json` and `counters.csv` | `v2_gpu_utilisation_by_run`, `v2_gpu_utilisation_by_ecosystem` | Whether the accelerator was working. Energy and duration alone cannot distinguish a slow kernel from an idle card waiting on a host, which is exactly the question for the stack the study calls expensive. A run counts only if its whole window — `machine_state.utc` to the last `counters.csv` append — falls inside the record, and the script reports which runs those are rather than averaging over whatever happens to be there. |

`experiment_spec.md` is the specification the replicated campaign was run
against; `repetition_protocol.md` specifies how the repetitions must be
executed.

## What the audit half established

1. `energy_consumed` is **kWh**, not Joules — a factor of 3.6e6. Verified three
   ways in `01_data_audit.py`.
2. The submitted campaign contains **2,880** tracked blocks (1,440 training +
   1,440 inference) and **one** run per configuration, not 7,200.
3. The instrument was **not the same** across ecosystems: two CodeCarbon major
   versions, two tracking modes, two sampling intervals. For the CodeCarbon
   2.8.4 stacks roughly two thirds of the reported energy is a constant modelled
   host power times duration. Hence the GPU-only and harmonised boundaries in
   `common.py`.
4. The training protocol differed between stacks in ways the manuscript said it
   did not — learning rate, loader parallelism, input scaling — which `10`
   documents line by line and the revision aligned.

Everything downstream of that is the re-executed campaign, and it is what the
manuscript now reports.
