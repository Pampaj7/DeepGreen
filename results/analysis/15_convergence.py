#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training collapse, and what it does to a fixed-budget energy comparison.

Replicating each configuration five times surfaced something a single-run design
cannot see: VGG-16 does not always train. In a fraction of runs it converges to
exactly chance accuracy -- 1/100 on CIFAR-100, 1/200 on Tiny ImageNet -- and
stays there, having consumed the full epoch budget of energy while learning
nothing.

The collapse is a property of the recipe, not of any ecosystem. It happens to
VGG-16 and never to ResNet-18, on the many-class datasets and never on
Fashion-MNIST, and it happens in several ecosystems independently. VGG-16 as
shipped has no batch normalisation, and a plain 16-layer network initialised
randomly and driven by Adam at 1e-4 across 100 or 200 classes can settle into
predicting a single class; whether it does is decided by the seed.

Two consequences matter for this study.

First, it is the strongest available evidence that the specification worked.
Conditional on converging, the ecosystems reach accuracies within a couple of
percentage points of one another. The apparent cross-stack accuracy spread in
the raw numbers is almost entirely these collapsed runs.

Second, it breaks fixed-budget energy comparison in a way that is invisible
without accuracy. A collapsed run costs full energy and produces nothing, so an
ecosystem that drew unlucky seeds looks equally expensive and much less
accurate, and one that drew lucky seeds looks efficient. With one run per
configuration -- the submitted design -- the reported result is whichever
outcome that single seed produced.

Writes results/revision/tables/v2_convergence_*.{md,csv}.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, TABLES_RESOLVER,  # noqa: E402
                    freeman_halton_exact, read_campaign_metrics, save_table)

TABLES = TABLES_RESOLVER  # writes divert on a live campaign, reads fall back

# Which campaign this run analyses. The collapse finding belongs to the first
# campaign -- the second has none -- and the manuscript now presents it as that
# campaign's result, so this script has to be able to produce both. The outputs
# are named apart (v1_convergence_*, v2_convergence_*) so that running one can
# never overwrite the other's tables, which is the same rule 23 of the revision
# log arrived at for partial campaigns.
CAMPAIGNS = {
    "v2": REPO_ROOT / "results" / "campaign_v2",
    "v1": REPO_ROOT / "results" / "campaign_v2_first_campaign",
}

# Accuracy of a classifier that always predicts one class.
CHANCE_PCT = {"fashionmnist": 100 / 10, "cifar100": 100 / 100,
              "tinyimagenet": 100 / 200}
# A run counts as collapsed if it never exceeded 1.5x chance. The multiplier is
# generous on purpose: every collapsed run in this campaign sits *exactly* at
# chance, so no borderline case depends on where the line is drawn.
COLLAPSE_FACTOR = 1.5

# The columns of <prefix>_convergence_signature, named here so that a campaign
# nothing to report still writes the header. See main() for why that matters.
SIGNATURE_COLUMNS = ["run", "ecosystem", "model", "dataset", "final_test_acc_pct",
                     "train_loss_drop_pct", "test_loss_first", "test_loss_last",
                     "diagnosis"]
HOMOGENEITY_COLUMNS = ["ecosystem", "n_collapsed", "n_runs", "collapse_pct",
                       "overall_pct", "exact_p", "permutation_p", "chi_square",
                       "chi_square_p", "chi_square_min_expected"]
CONDITIONAL_COLUMNS = ["model", "dataset", "n_ecosystems", "raw_spread_pp",
                       "converged_spread_pp", "converged_min_pct",
                       "converged_max_pct"]

#: The quantities load() must have as floats.
NUMERIC = ("final_test_acc_pct", "train_energy_total_J", "acc_per_kJ",
           "target_acc_pct", "energy_to_target_J")


def campaign_module():
    """09's collector and quality normalisation, imported rather than copied."""
    path = Path(__file__).resolve().parent / "09_campaign_v2.py"
    spec = importlib.util.spec_from_file_location("09_campaign_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(prefix: str, campaign_dir: Path) -> pd.DataFrame:
    """The per-run quality table for this campaign.

    For the current campaign that is the table 09 writes. For the superseded one
    there is no such table and there should not be -- 09 aggregates the campaign
    the paper reports -- so it is computed here from 09's own collector with the
    root parameterised, the way 18_precision_ablation reads the same directory.
    """
    path = TABLES / f"{prefix}_quality_normalised.csv"
    if path.exists():
        q = pd.read_csv(path)
    else:
        nine = campaign_module()
        q = nine.quality_normalised(nine.collect(campaign_dir)).round(3)
    # A header-only CSV -- which 09 now writes when a campaign has produced no
    # quality rows yet -- reads back with object columns, and every arithmetic
    # below then behaves differently from the populated case: sum() returns an
    # int 0, so the waste share raises ZeroDivisionError rather than being zero.
    # Coerce once here instead of defending against dtype in five places.
    for column in NUMERIC:
        if column in q:
            q[column] = q[column].astype(float)
    q["chance_pct"] = q.dataset.map(CHANCE_PCT).astype(float)
    q["collapsed"] = q.final_test_acc_pct <= q.chance_pct * COLLAPSE_FACTOR
    return q


def collapse_signature(campaign_dir: Path) -> pd.DataFrame:
    """Distinguish a failed recipe from a broken pipeline.

    Both produce chance test accuracy, and in an energy table they are
    indistinguishable. The per-epoch traces separate them cleanly:

      * optimisation collapse -- the network settles on one class. Training loss
        stalls too, because there is nothing to fit. This is a property of the
        recipe and happens to every stack that draws an unlucky initialisation.

      * pipeline defect -- the network fits the training data perfectly while
        test accuracy sits at chance and test loss *rises*. Train and test are
        not seeing the same inputs.

    The second is how we found that one Rust binary resized the tensor its
    loader had already produced, with a function returning uint8, so every value
    in [0,1] truncated to zero: it trained on black images and was evaluated on
    real ones. Training loss fell from 1.13 to 0.15 while accuracy stayed at
    chance for thirty epochs. Nothing in the energy data showed it, and the
    collapse-rate table alone would have filed it under "VGG-16 sometimes fails
    to train".

    The traces come from common.read_campaign_metrics, which is also what
    12_paper_numbers's collapse_mechanism_facts and check_consistency's S5 read.
    This walked the run directories itself with no completeness gate, while 12
    counted the same phenomenon behind one, so \\vSigCollapseN and
    \\vCollapseRuns could describe different populations of the same campaign
    and nothing would say which. They are both zero today, which is exactly how
    a divergence like that stays invisible until it matters.
    """
    rows: list[dict] = []
    metrics = read_campaign_metrics(root=campaign_dir)
    if metrics.empty or not {"test_acc", "train_loss"}.issubset(metrics.columns):
        return pd.DataFrame(rows, columns=SIGNATURE_COLUMNS)
    for run, m in metrics.groupby("run"):
        m = m.sort_values("epoch")
        dataset = str(m.dataset.iat[0])
        chance = CHANCE_PCT.get(dataset)
        if chance is None:
            continue
        final_acc = float(m.test_acc.iat[-1])
        if final_acc > chance * COLLAPSE_FACTOR:
            continue
        first, last = float(m.train_loss.iat[0]), float(m.train_loss.iat[-1])
        loss_drop = (first - last) / first if first else 0.0
        test_first = float(m.test_loss.iat[0]) if "test_loss" in m else float("nan")
        test_last = float(m.test_loss.iat[-1]) if "test_loss" in m else float("nan")
        rows.append({
            "run": run,
            "ecosystem": str(m.ecosystem.iat[0]), "model": str(m.model.iat[0]),
            "dataset": dataset,
            "final_test_acc_pct": round(final_acc, 2),
            "train_loss_drop_pct": round(100 * loss_drop, 1),
            "test_loss_first": round(test_first, 2),
            "test_loss_last": round(test_last, 2),
            # Fitting the training set while failing the test set is not an
            # optimisation failure.
            "diagnosis": ("pipeline defect: fits train, fails test"
                          if loss_drop > 0.5 and test_last >= test_first
                          else "optimisation collapse"),
        })
    return pd.DataFrame(rows, columns=SIGNATURE_COLUMNS)


def by_model(q: pd.DataFrame) -> pd.DataFrame:
    t = (q.groupby(["model", "dataset"])
         .agg(n_runs=("collapsed", "size"), n_collapsed=("collapsed", "sum"))
         .reset_index())
    t["collapse_pct"] = (100 * t.n_collapsed / t.n_runs).round(1)
    return t


def by_ecosystem(q: pd.DataFrame) -> pd.DataFrame:
    v = q[q.model == "vgg16"]
    t = (v.groupby(["ecosystem", "dataset"])
         .agg(n_runs=("collapsed", "size"), n_collapsed=("collapsed", "sum"))
         .reset_index())
    # Ecosystems with zero collapses are reported rather than filtered out:
    # "JAX never collapsed" is as much a result as "Java collapsed five times".
    return t.sort_values(["dataset", "ecosystem"])


def homogeneity_test(q: pd.DataFrame, n_permutations: int = 20000) -> pd.DataFrame:
    """Is the collapse rate the same in every ecosystem?

    Restricted to the cells where collapse occurs at all (VGG-16 on the
    many-class datasets), so the question is conditional on susceptibility
    rather than confounded with it.

    The table is seven ecosystems by ten runs with twelve collapses in total.
    Chi-square wants five expected observations per cell and has 1.71, so its
    p-value is not to be quoted. The answer here is the exact one:
    Freeman--Halton, the r x c generalisation of Fisher's test, conditions on
    both margins and sums the probability of every table no more likely than
    this one. A few thousand terms, no approximation, no seed.

    The permutation test is kept beside it, and the manuscript should say why
    the two differ. Its statistic is ``max(rate) - min(rate)``, which sees only
    the extremes: it cannot tell seven ecosystems at 0, 0, 0, 10, 20, 40 and 50
    per cent from two at 0 and 50 with five in between at the mean. That is why
    it returns a larger p-value than either the exact test or chi-square, and it
    is a property of the statistic rather than evidence about the data.

    Chi-square is reported too, because the manuscript contrasts the exact and
    approximate answers and a contrast is only honest if both sides come from
    this table. It is guarded: with a zero expected frequency -- which a partial
    campaign produces -- scipy raises, and a test we are arguing against must
    not be able to stop the pipeline.
    """
    v = q[(q.model == "vgg16") & (q.dataset.isin(["cifar100", "tinyimagenet"]))]
    if v.empty:
        # No susceptible cell in this campaign: there is no table to test and
        # no statistic that means anything over one. The permutation statistic
        # would reduce max() over an empty array, and Freeman--Halton would
        # index past the end of a zero-group margin. Return the header and let
        # main() write it -- see there for why an empty table still has to be
        # written.
        return pd.DataFrame(columns=HOMOGENEITY_COLUMNS)
    table = v.groupby("ecosystem").collapsed.agg(["sum", "size"])

    def spread(labels: np.ndarray, groups: np.ndarray) -> float:
        rates = [labels[groups == g].mean() for g in np.unique(groups)]
        return float(np.max(rates) - np.min(rates))

    labels = v.collapsed.to_numpy()
    groups = v.ecosystem.to_numpy()
    observed = spread(labels, groups)
    rng = np.random.default_rng(20260827)
    extreme = sum(spread(rng.permutation(labels), groups) >= observed
                  for _ in range(n_permutations))
    p_value = (extreme + 1) / (n_permutations + 1)

    rows = table.reset_index().rename(columns={"sum": "n_collapsed", "size": "n_runs"})
    rows["collapse_pct"] = (100 * rows.n_collapsed / rows.n_runs).round(0)
    rows["overall_pct"] = round(100 * labels.mean(), 1)

    contingency = np.column_stack([table["sum"].to_numpy(),
                                   (table["size"] - table["sum"]).to_numpy()])

    exact_p, _ = freeman_halton_exact(contingency)
    rows["exact_p"] = round(exact_p, 4)
    rows["permutation_p"] = round(p_value, 4)

    try:
        chi2, chi_p, _, expected = stats.chi2_contingency(contingency)
        rows["chi_square"] = round(float(chi2), 3)
        rows["chi_square_p"] = round(float(chi_p), 4)
        rows["chi_square_min_expected"] = round(float(expected.min()), 2)
    except ValueError:
        # a zero expected frequency: exactly the regime the exact test is for
        rows["chi_square"] = np.nan
        rows["chi_square_p"] = np.nan
        rows["chi_square_min_expected"] = 0.0
    return rows


def conditional_accuracy(q: pd.DataFrame) -> pd.DataFrame:
    """Accuracy among runs that trained at all -- the like-for-like comparison."""
    rows = []
    for (model, dataset), g in q.groupby(["model", "dataset"]):
        ok = g[~g.collapsed]
        per_eco = ok.groupby("ecosystem").final_test_acc_pct.mean()
        raw = g.groupby("ecosystem").final_test_acc_pct.mean()
        rows.append({
            "model": model, "dataset": dataset,
            "n_ecosystems": len(per_eco),
            "raw_spread_pp": round(raw.max() - raw.min(), 2),
            "converged_spread_pp": round(per_eco.max() - per_eco.min(), 2),
            "converged_min_pct": round(per_eco.min(), 2),
            "converged_max_pct": round(per_eco.max(), 2),
        })
    return pd.DataFrame(rows, columns=CONDITIONAL_COLUMNS)


def wasted_energy(q: pd.DataFrame) -> pd.DataFrame:
    """Energy spent by runs that learned nothing."""
    total = float(q.train_energy_total_J.sum())
    wasted = float(q[q.collapsed].train_energy_total_J.sum())
    return pd.DataFrame([{
        "n_runs": len(q),
        "n_collapsed": int(q.collapsed.sum()),
        # Training energy only: the collapsed runs waste a training budget, and
        # quoting them against a train+eval total would flatter the figure.
        "training_energy_MJ": round(total / 1e6, 2),
        "collapsed_energy_MJ": round(wasted / 1e6, 2),
        # No training energy at all means no share of it wasted. Zero is the
        # answer, not a division to defend -- and n_runs beside it says the
        # campaign is empty, so the row cannot be mistaken for a result.
        "wasted_pct": round(100 * wasted / total, 1) if total else 0.0,
    }])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--campaign", choices=sorted(CAMPAIGNS), default="v2",
                    help="v2 (default) is the campaign the paper reports; v1 is "
                         "the superseded one, which is where the collapses are")
    args = ap.parse_args(argv)
    prefix, campaign_dir = args.campaign, CAMPAIGNS[args.campaign]
    if not campaign_dir.is_dir():
        print(f"no campaign under {campaign_dir.relative_to(REPO_ROOT)}")
        return
    print(f"campaign {prefix}: {campaign_dir.relative_to(REPO_ROOT)} "
          f"-> {prefix}_convergence_*")

    # Every one of the six tables below is written, populated or not. Three of
    # them used not to be: with an empty quality table -- the early-monitoring
    # case tables_dir() exists to serve -- homogeneity_test reduced max() over
    # an empty array and wasted_energy divided by zero, so the script died after
    # writing three tables and before writing the other three, and
    # 12_paper_numbers read those three through a resolver that falls back to
    # the committed copy. A partial run then quoted the FIRST campaign's
    # homogeneity p, conditional spread and wasted share, silently. Refusing to
    # write is not a safe failure here; it is the failure.
    q = load(prefix, campaign_dir)

    bm = by_model(q)
    print("--- collapse rate by model and dataset ---")
    print(bm.to_string(index=False))
    save_table(bm, f"{prefix}_convergence_by_model",
               "Runs that never exceeded chance accuracy, by model and dataset")

    be = by_ecosystem(q)
    print("\n--- which ecosystems saw it (VGG-16 only) ---")
    print(be.to_string(index=False))
    save_table(be, f"{prefix}_convergence_by_ecosystem",
               "VGG-16 collapses per ecosystem; the effect is not stack-specific")

    sig = collapse_signature(campaign_dir)
    if len(sig):
        print("\n--- failed recipe, or broken pipeline? ---")
        print(sig[["run", "final_test_acc_pct", "train_loss_drop_pct",
                   "test_loss_first", "test_loss_last", "diagnosis"]]
              .to_string(index=False))
    else:
        print("\n--- no run finished at chance accuracy ---")
    # Written even when empty. This was guarded by `if len(sig)`, and with zero
    # collapses the table was simply not written -- so 12_paper_numbers's
    # `.exists()` check resolved to the first campaign's committed copy and
    # emitted \vSigCollapseN from twelve runs that are no longer in the
    # campaign. A table this script owns has to exist for every campaign it
    # analyses, header and no rows when there is nothing to report, or a
    # downstream existence check silently means "the previous campaign".
    save_table(sig, f"{prefix}_convergence_signature",
               "Chance-accuracy runs separated by their per-epoch traces")
    broken = sig[sig.diagnosis.str.startswith("pipeline")]
    if len(broken):
        print(f"\n  !! {len(broken)} run(s) show a pipeline defect, "
              f"not an optimisation failure:")
        for r in broken.run:
            print(f"     {r}")

    ht = homogeneity_test(q)
    print("\n--- is the collapse rate the same in every ecosystem? ---")
    print(ht.to_string(index=False))
    save_table(ht, f"{prefix}_convergence_homogeneity",
               "VGG-16 collapse rate per ecosystem, with a permutation test of homogeneity")

    ca = conditional_accuracy(q)
    print("\n--- cross-ecosystem accuracy spread, raw against converged-only ---")
    print(ca.to_string(index=False))
    save_table(ca, f"{prefix}_convergence_conditional",
               "Cross-ecosystem accuracy spread before and after excluding collapsed runs")

    we = wasted_energy(q)
    print("\n--- energy spent on runs that learned nothing ---")
    print(we.to_string(index=False))
    save_table(we, f"{prefix}_convergence_waste",
               "Share of campaign energy consumed by collapsed runs")


if __name__ == "__main__":
    main()
