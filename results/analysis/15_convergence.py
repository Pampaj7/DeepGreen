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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, save_table  # noqa: E402

TABLES = REPO_ROOT / "results" / "revision" / "tables"

# Accuracy of a classifier that always predicts one class.
CHANCE_PCT = {"fashionmnist": 100 / 10, "cifar100": 100 / 100,
              "tinyimagenet": 100 / 200}
# A run counts as collapsed if it never exceeded 1.5x chance. The multiplier is
# generous on purpose: every collapsed run in this campaign sits *exactly* at
# chance, so no borderline case depends on where the line is drawn.
COLLAPSE_FACTOR = 1.5


def load() -> pd.DataFrame:
    q = pd.read_csv(TABLES / "v2_quality_normalised.csv")
    q["chance_pct"] = q.dataset.map(CHANCE_PCT)
    q["collapsed"] = q.final_test_acc_pct <= q.chance_pct * COLLAPSE_FACTOR
    return q


def collapse_signature() -> pd.DataFrame:
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
    """
    campaign = REPO_ROOT / "results" / "campaign_v2"
    rows = []
    for run_dir in sorted(p for p in campaign.glob("*") if p.is_dir()):
        mpath = run_dir / "metrics.csv"
        if not mpath.exists():
            continue
        try:
            m = pd.read_csv(mpath)
        except pd.errors.EmptyDataError:
            continue
        if m.empty or "test_acc" not in m or "train_loss" not in m:
            continue
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
            "run": run_dir.name,
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
    return pd.DataFrame(rows)


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

    Cell counts are far too small for the chi-square approximation, so the
    statistic is calibrated by permutation: shuffle the collapse labels across
    ecosystems many times and ask how often chance produces a spread at least as
    uneven as the observed one.
    """
    v = q[(q.model == "vgg16") & (q.dataset.isin(["cifar100", "tinyimagenet"]))]
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
    rows["permutation_p"] = round(p_value, 4)

    # The chi-square the paper warns against, computed rather than quoted: the
    # manuscript contrasts the two p-values, and a contrast is only honest if
    # both sides come from this table.
    contingency = np.column_stack([table["sum"].to_numpy(),
                                   (table["size"] - table["sum"]).to_numpy()])
    chi2, chi_p, _, expected = stats.chi2_contingency(contingency)
    rows["chi_square"] = round(float(chi2), 3)
    rows["chi_square_p"] = round(float(chi_p), 4)
    rows["chi_square_min_expected"] = round(float(expected.min()), 2)
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
    return pd.DataFrame(rows)


def wasted_energy(q: pd.DataFrame) -> pd.DataFrame:
    """Energy spent by runs that learned nothing."""
    total = q.train_energy_total_J.sum()
    wasted = q[q.collapsed].train_energy_total_J.sum()
    return pd.DataFrame([{
        "n_runs": len(q),
        "n_collapsed": int(q.collapsed.sum()),
        # Training energy only: the collapsed runs waste a training budget, and
        # quoting them against a train+eval total would flatter the figure.
        "training_energy_MJ": round(total / 1e6, 2),
        "collapsed_energy_MJ": round(wasted / 1e6, 2),
        "wasted_pct": round(100 * wasted / total, 1),
    }])


def main() -> None:
    q = load()

    bm = by_model(q)
    print("--- collapse rate by model and dataset ---")
    print(bm.to_string(index=False))
    save_table(bm, "v2_convergence_by_model",
               "Runs that never exceeded chance accuracy, by model and dataset")

    be = by_ecosystem(q)
    print("\n--- which ecosystems saw it (VGG-16 only) ---")
    print(be.to_string(index=False))
    save_table(be, "v2_convergence_by_ecosystem",
               "VGG-16 collapses per ecosystem; the effect is not stack-specific")

    sig = collapse_signature()
    if len(sig):
        print("\n--- failed recipe, or broken pipeline? ---")
        print(sig[["run", "final_test_acc_pct", "train_loss_drop_pct",
                   "test_loss_first", "test_loss_last", "diagnosis"]]
              .to_string(index=False))
        save_table(sig, "v2_convergence_signature",
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
    save_table(ht, "v2_convergence_homogeneity",
               "VGG-16 collapse rate per ecosystem, with a permutation test of homogeneity")

    ca = conditional_accuracy(q)
    print("\n--- cross-ecosystem accuracy spread, raw against converged-only ---")
    print(ca.to_string(index=False))
    save_table(ca, "v2_convergence_conditional",
               "Cross-ecosystem accuracy spread before and after excluding collapsed runs")

    we = wasted_energy(q)
    print("\n--- energy spent on runs that learned nothing ---")
    print(we.to_string(index=False))
    save_table(we, "v2_convergence_waste",
               "Share of campaign energy consumed by collapsed runs")


if __name__ == "__main__":
    main()
