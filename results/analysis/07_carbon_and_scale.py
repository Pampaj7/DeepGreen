#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Campaign carbon footprint and the fleet-scale illustration.

Addresses:
  R1 c11 / R3 M6 -- the 100,000-GPU extrapolation asserts more than the data
                    supports and mixes energy boundaries. Recomputed with the
                    boundaries made explicit and reframed as a bounded
                    illustration for the measured model class only.
  R3 M7          -- carbon recomputed over the true measurement count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import J_PER_KWH, load, save_table

GRID_INTENSITY_KG_PER_KWH = 0.3307  # CodeCarbon, Tuscany, at campaign time


def main() -> None:
    df = load()
    print("=" * 78)
    print("CARBON FOOTPRINT AND FLEET-SCALE ILLUSTRATION  (R1 c11, R3 M6/M7)")
    print("=" * 78)

    total_kwh = df["energy_j"].sum() / J_PER_KWH
    total_kg = df["emissions_kg"].sum()
    print(f"\n  measurement blocks           : {len(df)}")
    print(f"  total measured energy        : {total_kwh:.2f} kWh ({df['energy_j'].sum()/1e6:.1f} MJ)")
    print(f"  total measured emissions     : {total_kg:.2f} kg CO2eq")
    print(f"  implied grid intensity       : {1000*total_kg/total_kwh:.0f} gCO2eq/kWh")
    print(
        "\n  These figures cover the tracked blocks only. They exclude compilation,\n"
        "  dataset preparation, idle time between runs and any failed or discarded runs,\n"
        "  so they are a lower bound on the campaign's true footprint."
    )
    save_table(
        pd.DataFrame(
            {
                "quantity": [
                    "measurement blocks",
                    "training blocks",
                    "inference blocks",
                    "total energy (kWh)",
                    "total emissions (kg CO2eq)",
                    "grid intensity (gCO2eq/kWh)",
                ],
                "value": [
                    len(df),
                    int((df["phase"] == "Training").sum()),
                    int((df["phase"] == "Inference").sum()),
                    round(total_kwh, 3),
                    round(total_kg, 3),
                    round(1000 * total_kg / total_kwh, 1),
                ],
            }
        ),
        "campaign_carbon_footprint",
        "Carbon footprint of the tracked measurement blocks",
    )

    # --- fleet illustration, boundaries made explicit -----------------------
    print("\n--- fleet-scale illustration, with the boundary stated ---")
    inf = df[df["phase"] == "Inference"]
    per_eco = inf.groupby("ecosystem", observed=True).agg(
        energy_as_measured_J=("energy_j", "mean"),
        energy_harmonised_J=("energy_harm_j", "mean"),
        energy_gpu_J=("energy_gpu_j", "mean"),
    )
    best_h = per_eco["energy_harmonised_J"].idxmin()
    worst_h = per_eco["energy_harmonised_J"].idxmax()

    rows = []
    for label, col in [
        ("as measured (CPU+GPU+RAM, mixed instruments)", "energy_as_measured_J"),
        ("harmonised (GPU + uniform 107 W host)", "energy_harmonised_J"),
        ("GPU only (comparable to a per-GPU power budget)", "energy_gpu_J"),
    ]:
        ratio = per_eco[col].max() / per_eco[col].min()
        rows.append({"boundary": label, "worst_over_best_x": ratio})
    bt = pd.DataFrame(rows).round(2)
    print(bt.to_string(index=False))
    print(
        f"\n  The manuscript multiplies an assumed per-GPU board power by a ratio derived\n"
        "  from CPU+GPU+RAM energy. Those are different boundaries, and the resulting\n"
        "  figure is not a like-for-like saving. The only ratio that may legitimately\n"
        "  be applied to a per-GPU power budget is the GPU-only one "
        f"({bt.loc[2,'worst_over_best_x']:.2f}x).\n"
        f"  Best/worst inference ecosystems under the harmonised boundary: "
        f"{best_h} / {worst_h}."
    )
    save_table(bt, "fleet_boundary_sensitivity", "Ratio applied in the fleet illustration, by energy boundary")

    print(
        "\n  RECOMMENDATION: the fleet projection assumes the measured ratio transfers to\n"
        "  a different model family, operator mix, batch regime, precision policy and\n"
        "  serving stack -- none of which this campaign varied, and all of which the\n"
        "  audit shows dominate the measured quantity. It should be moved to an\n"
        "  appendix, restricted to ResNet-18/VGG-16 at 32x32 on a single L40S, stated\n"
        "  in GPU energy only, and given as a range rather than a point estimate."
    )


if __name__ == "__main__":
    main()
