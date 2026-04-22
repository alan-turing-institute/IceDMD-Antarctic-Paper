#!/usr/bin/env python
"""
Figure 8 -- Combined panel has two parts: spatial SIC maps + probe time series.

These components are produced separately and stiched together for the publication.

Layout (3 rows x 3 columns):
(a-c)  Observed SIC at three dates (Mar, Jun, Sep 2023)
(d-f)  DMD-predicted SIC at the same dates
(g-i)  Time series at three probe locations (A, B, C) with
       observations in red, DMD mean in black, and +/- 2 sigma with physical bounds (0-1)
       uncertainty in grey.  Small inset maps show the probe positions.

Generates: figures/figure_8_prediction_part1.png, figures/figure_8_prediction_part2.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE, PROBES, YEAR_INDEX
from sic_dmd.plotting import (
    day_index_to_date, day_of_year,
    plot_antarctic_map, plot_probe_inset,
)


def main():
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    X_true = r["X_test_true"]
    X_pred = r["X_test_mean_filt"]
    mask_land = r["mask_land"]
    probe_test = r["probe_test"]
    probe_obs = r["probe_obs"]

    # Dates for the spatial snapshots
    snapshot_days = [
        (day_of_year(3, 1), "01 Mar 2023"),
        (day_of_year(6, 1), "01 Jun 2023"),
        (day_of_year(9, 1), "01 Sep 2023"),
    ]

    n_days = 365
    dates = [day_index_to_date(t) for t in range(n_days)]

    # -- Build the first part of the figure (the spatial maps at given times) --
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels_top = list("abc")
    labels_bot = list("def")
    for col, (day_idx, title) in enumerate(snapshot_days):
        # Observed
        ax_obs = axes[0, col]
        plot_antarctic_map(ax_obs, X_true[day_idx], mask_land, title=title)
        ax_obs.text(0.02, 0.95, f"({labels_top[col]})", transform=ax_obs.transAxes,
                    fontsize=11, va="top", fontweight="bold", color="w")
        if col == 0:
            ax_obs.set_ylabel("Observed", fontsize=13, labelpad=8)

        # Predicted
        ax_pred = axes[1, col]
        plot_antarctic_map(ax_pred, np.clip(X_pred[day_idx], 0, 1), mask_land)
        ax_pred.text(0.02, 0.95, f"({labels_bot[col]})", transform=ax_pred.transAxes,
                     fontsize=11, va="top", fontweight="bold", color="w")
        if col == 0:
            ax_pred.set_ylabel("Predicted (DMD)", fontsize=13, labelpad=8)
    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_8_prediction_part1.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # -- Build the 2nd part of the figure (time series at individual points) --
    fig = plt.figure(figsize=(12, 7))

    # --- Probe time series (row g-i) ---
    gs_probes = gridspec.GridSpec(
        3, 2, wspace=0.05, width_ratios=[5, 1.2], hspace=0.05
    )
    labels_probe = list("ghi")
    for row, (name, pt) in enumerate(PROBES.items()):
        # Each probe panel: time series on the left, small map inset on the right
        ax_ts = fig.add_subplot(gs_probes[row, 0])
        ax_map = fig.add_subplot(gs_probes[row, 1])

        # Vertical lines corresponding to the spatial map times
        for col, (day_idx, title) in enumerate(snapshot_days):
            ax_ts.axvline(dates[day_idx], color='0.5', ls='--', lw=0.5)

        # Observations for 2023
        obs = probe_obs[name][YEAR_INDEX][:n_days]
        ax_ts.plot(dates[:len(obs)], obs, color="red", lw=0.8)

        # DMD ensemble mean +/- 2 sigma
        ens = probe_test[name][:, :n_days]
        m = ens.mean(axis=0)
        s = ens.std(axis=0)

        # Bound to physical values
        min_uq = m - 2 * s
        min_uq[min_uq < 0] = 0
        max_uq = m + 2 * s
        max_uq[max_uq > 1] = 1
        ax_ts.plot(dates, m, color="black", lw=1.2)
        ax_ts.fill_between(dates, min_uq, max_uq,
                           color="black", alpha=0.1)

        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.set_ylabel("SIC")
        ax_ts.text(0.02, 0.95, f"({labels_probe[row]})",
                   transform=ax_ts.transAxes, fontsize=11, va="top",
                   fontweight="bold")
        ax_ts.set_xlim(dates[0], dates[-1])
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_ts.tick_params(axis="x", rotation=30)
        if row < 2:
            ax_ts.set_xticklabels([])

        # Inset map
        plot_probe_inset(ax_map, mask_land, pt, name)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_8_prediction_part2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
