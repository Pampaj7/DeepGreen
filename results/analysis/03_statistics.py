#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical basis for the ecosystem comparison.

Addresses:
  R1 c5  / R3 M2 -- inferential tests, effect sizes, confidence intervals,
                    and an explicit statement of what the design can support
  R1 c3          -- the four LibTorch stacks as a shared-backend control group
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import cliffs_delta, load, save_table

PSEUDO_REPLICATION_NOTE = """
IMPORTANT -- what these tests can and cannot establish.

Each of the 48 configurations was executed exactly ONCE. The 30 epochs of a run
are repeated measurements of the same run, not independent replications: they
share one process, one weight initialisation, one allocator state, one JIT
outcome and one thermal trajectory. The effective number of independent
observations per configuration is therefore 1, not 30.

The tests below are computed over epochs and are reported for completeness and
comparability with the prior literature, but their p-values are anti-conservative
by an unknown factor: they test whether the epochs of run A differ from the
epochs of run B, not whether ecosystem A differs from ecosystem B. Effect sizes
(Cliff's delta) and the observed dispersion are the more informative outputs.

Any claim of a statistically significant *ecosystem* difference requires
independent run-level repetitions. results/analysis/repetition_protocol.md
specifies the protocol; scripts/run_campaign.py implements it.
"""


def omnibus(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for phase, sub in df.groupby("phase", observed=True):
        groups = [g[col].to_numpy() for _, g in sub.groupby("ecosystem", observed=True)]
        h, p = stats.kruskal(*groups)
        # epsilon-squared effect size for Kruskal-Wallis
        n = sum(len(g) for g in groups)
        k = len(groups)
        eps2 = (h - k + 1) / (n - k)
        rows.append(
            {
                "phase": phase,
                "metric": col,
                "test": "Kruskal-Wallis",
                "H": h,
                "df": k - 1,
                "p_value": p,
                "epsilon_squared": eps2,
            }
        )
    return pd.DataFrame(rows)


def pairwise(df: pd.DataFrame, col: str, phase: str) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    ecos = sorted(sub["ecosystem"].unique())
    rows = []
    for a, b in itertools.combinations(ecos, 2):
        xa = sub.loc[sub["ecosystem"] == a, col].to_numpy()
        xb = sub.loc[sub["ecosystem"] == b, col].to_numpy()
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        delta, mag = cliffs_delta(xa, xb)
        rows.append(
            {
                "phase": phase,
                "a": a,
                "b": b,
                "median_a_J": float(np.median(xa)),
                "median_b_J": float(np.median(xb)),
                "ratio_b_over_a": float(np.median(xb) / np.median(xa)),
                "U": u,
                "p_raw": p,
                "cliffs_delta": delta,
                "magnitude": mag,
            }
        )
    out = pd.DataFrame(rows)
    # Holm-Bonferroni over the family of pairwise comparisons within the phase
    order = np.argsort(out["p_raw"].to_numpy())
    m = len(out)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * out["p_raw"].iloc[idx]
        running = max(running, min(val, 1.0))
        adj[idx] = running
    out["p_holm"] = adj
    out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def libtorch_control(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """R1 comment 3: the LibTorch stacks share a backend, so any spread between
    them is host-side (binding, data loading, dispatch, toolchain version)."""
    sub = df[df["backend"] == "LibTorch"]
    rows = []
    for phase, s in sub.groupby("phase", observed=True):
        m = s.groupby("ecosystem", observed=True)[col].mean()
        allm = df[df["phase"] == phase].groupby("ecosystem", observed=True)[col].mean()
        rows.append(
            {
                "phase": phase,
                "metric": col,
                "libtorch_stacks": ", ".join(sorted(m.index)),
                "libtorch_spread_x": m.max() / m.min(),
                "full_spread_x": allm.max() / allm.min(),
                "pct_of_full_spread": 100
                * np.log(m.max() / m.min())
                / np.log(allm.max() / allm.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print("=" * 78)
    print("STATISTICAL ANALYSIS  (R1 c5, R3 M2)")
    print("=" * 78)
    print(PSEUDO_REPLICATION_NOTE)

    frames = []
    for col in ["energy_j", "energy_harm_j", "duration_s"]:
        frames.append(omnibus(df, col))
    om = pd.concat(frames, ignore_index=True)
    print("--- omnibus tests across the eight ecosystems ---")
    print(om.to_string(index=False))
    save_table(
        om.round(6),
        "stats_omnibus",
        "Kruskal-Wallis across ecosystems (epoch-level; see pseudo-replication caveat)",
    )

    for phase in ["Training", "Inference"]:
        pw = pairwise(df, "energy_harm_j", phase)
        n_sig = int(pw["significant_0.05"].sum())
        n_large = int((pw["magnitude"] == "large").sum())
        print(
            f"\n--- pairwise, {phase} (harmonised energy): "
            f"{n_sig}/{len(pw)} significant after Holm, {n_large}/{len(pw)} large effect ---"
        )
        print(
            pw.sort_values("p_holm")
            .head(8)[["a", "b", "ratio_b_over_a", "p_holm", "cliffs_delta", "magnitude"]]
            .round(4)
            .to_string(index=False)
        )
        save_table(
            pw.round(6),
            f"stats_pairwise_{phase.lower()}",
            f"{phase}: pairwise Mann-Whitney with Holm correction and Cliff's delta "
            "on harmonised energy (epoch-level pseudo-replicates)",
        )

    print("\n--- shared-backend control: the four LibTorch stacks (R1 comment 3) ---")
    lt = pd.concat(
        [libtorch_control(df, c) for c in ["energy_j", "energy_harm_j", "duration_s"]],
        ignore_index=True,
    ).round(2)
    print(lt.to_string(index=False))
    save_table(lt, "stats_libtorch_control", "Spread within the shared-backend LibTorch group")
    print(
        "\nINTERPRETATION: four of the eight stacks (Rust/tch, C++/LibTorch,\n"
        "  Python/PyTorch, R/torch) run the same LibTorch kernels. They still span a\n"
        "  large share of the total spread, which shows directly that the measured\n"
        "  differences are host-side -- binding overhead, data loading, dispatch and\n"
        "  toolchain version -- rather than a property of the language. The\n"
        "  contribution should be framed as binding/runtime/toolchain overhead, not\n"
        "  as language energy efficiency."
    )

    with open("../revision/tables/stats_caveat.md", "w") as fh:
        fh.write("# Pseudo-replication caveat\n" + PSEUDO_REPLICATION_NOTE)


if __name__ == "__main__":
    main()
