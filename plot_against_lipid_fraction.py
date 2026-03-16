#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:37:15 2026

@author: brunokeyworth
"""


import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=100)
plt.rcParams.update({
    "font.size": 14,          # base size
    "axes.titlesize": 16,     # subplot titles
    "axes.labelsize": 14,     # x and y labels
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

colours = {
    "Control": "black",
    "C12E6": "tab:blue",
    "DDAC": "tab:red",
    "TX100": "tab:green",
}

peak_labels = {
    "pos": "Mean Peak Position (nm)",
    "width": "Peak Width (nm)",
    "area": "Peak Area (%)",
}

def mean_err(entry):
    if entry and isinstance(entry, list) and len(entry) == 2:
        return entry[0], entry[1]
    return np.nan, np.nan


def gather_data(folder):

    records = []

    for fp in folder.glob("*.json"):
        with open(fp) as f:
            data = json.load(f)

        if not np.isclose(data.get("lipid_conc_mg_ml", np.nan), 0.1):
            continue

        lipid = data["lipid_ratio"]
        surf = data["surfactant_conc_microM"]

        DMPC = lipid.get("DMPC", 0)
        DMPG = lipid.get("DMPG", 0)

        if DMPC == 0 and DMPG == 0:
            continue

        total = DMPC + DMPG
        frac_DMPG = DMPG / total if total > 0 else np.nan

        surf_total = surf.get("C12E6", 0) + surf.get("DDAC", 0) + surf.get("TX100", 0)

        peaks = data.get("averaged_peaks", [])

        p1 = peaks[0] if len(peaks) > 0 else {}
        p2 = peaks[1] if len(peaks) > 1 else {}
        p3 = peaks[2] if len(peaks) > 2 else {}

        zeta, zeta_err = mean_err(data.get("average_zeta"))

        p1_pos, p1_pos_err = mean_err(p1.get("peak_position_nm"))
        p1_width, p1_width_err = mean_err(p1.get("peak_width_nm"))
        p1_area, p1_area_err = mean_err(p1.get("area_percent"))

        p2_pos, p2_pos_err = mean_err(p2.get("peak_position_nm"))
        p2_width, p2_width_err = mean_err(p2.get("peak_width_nm"))
        p2_area, p2_area_err = mean_err(p2.get("area_percent"))

        p3_pos, p3_pos_err = mean_err(p3.get("peak_position_nm"))
        p3_width, p3_width_err = mean_err(p3.get("peak_width_nm"))
        p3_area, p3_area_err = mean_err(p3.get("area_percent"))

        record = {
            "fraction_DMPG": frac_DMPG,
            "surfactant_total": surf_total,

            "zeta": zeta,
            "zeta_err": zeta_err,

            "p1_pos": p1_pos,
            "p1_pos_err": p1_pos_err,
            "p1_width": p1_width,
            "p1_width_err": p1_width_err,
            "p1_area": p1_area,
            "p1_area_err": p1_area_err,

            "p2_pos": p2_pos,
            "p2_pos_err": p2_pos_err,
            "p2_width": p2_width,
            "p2_width_err": p2_width_err,
            "p2_area": p2_area,
            "p2_area_err": p2_area_err,

            "p3_pos": p3_pos,
            "p3_pos_err": p3_pos_err,
            "p3_width": p3_width,
            "p3_width_err": p3_width_err,
            "p3_area": p3_area,
            "p3_area_err": p3_area_err,

            "C12E6": surf.get("C12E6", 0),
            "DDAC": surf.get("DDAC", 0),
            "TX100": surf.get("TX100", 0),
        }

        records.append(record)

    return pd.DataFrame(records)

def surfactant_condition(row):

    if row["C12E6"] == 0 and row["DDAC"] == 0 and row["TX100"] == 0:
        return "Control"

    if row["C12E6"] == 100:
        return "C12E6"
    if row["DDAC"] == 100:
        return "DDAC"
    if row["TX100"] == 100:
        return "TX100"

    return None

def create_concentration_plots(df, ratio=0.3):
    """
    Peak plots: 3 figures (pos, width, area), each with 3 subplots (one per surfactant).
    Control points appear at 0.1 µM on all subplots.
    Zeta plot: single figure for all surfactants
    """
    df = df[np.isclose(df["fraction_DMPG"], ratio)]
    surfactants = ["C12E6", "DDAC", "TX100"]
    peak_cols = ["pos", "width", "area"]

    # Identify control rows
    control = df[(df["C12E6"] == 0) & (df["DDAC"] == 0) & (df["TX100"] == 0)]
    
    for key in peak_cols:
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
        for i, surf in enumerate(surfactants):
    
            ax = axes[i]
            sub = df[df[surf] > 0].sort_values(surf)
    
            if sub.empty:
                continue
    
            rows = pd.concat([control, sub], ignore_index=True)
            conc = np.concatenate([np.full(len(control), 0), sub[surf].values])
    
            width = 0.25
    
            for k, (_, row) in enumerate(rows.iterrows()):
    
                peaks = []
                for peak in [1, 2, 3]:
                    peaks.append({
                        "value": row[f"p{peak}_{key}"],
                        "err": row[f"p{peak}_{key}_err"],
                        "area": row[f"p{peak}_area"]
                    })
    
                # Order by decreasing area
                peaks = sorted(peaks, key=lambda p: p["area"], reverse=True)
    
                for j, p in enumerate(peaks):
    
                    x = k + (j - 1) * width
                    colour = cmap(norm(p["area"]))
    
                    ax.bar(
                        x,
                        p["value"],
                        width=width,
                        yerr=p["err"],
                        capsize=3,
                        color=colour,
                        edgecolor="black",
                    )
    
            ax.set_xticks(range(len(conc)))
            ax.set_xticklabels([f"{c:g}" for c in conc])
            ax.set_xlabel(f"{surf} concentration (µM)")
            ax.set_ylabel(peak_labels[key])
            ax.set_title(surf)
    
        # Add shared colourbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cbar = fig.colorbar(sm, ax=axes, pad=0.02)
        cbar.set_label("Peak Area (%)")
    
        fig.savefig(PLOTS_FOLDER / f"{key}_conc_1x3.png", dpi=300)
        plt.show()

    # ----- Zeta plot (single plot for all surfactants) -----
    fig, ax = plt.subplots()
    for surf in surfactants:
        sub = df[df[surf] > 0].sort_values(surf)
        if sub.empty:
            continue

        x = np.concatenate([np.full(len(control), 0.1), sub[surf].values])
        y = np.concatenate([control["zeta"].values, sub["zeta"].values])
        yerr = np.concatenate([control["zeta_err"].values, sub["zeta_err"].values])

        ax.errorbar(
            x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, label=surf, color=colours[surf]
        )

    ax.set_xlabel("Surfactant concentration (µM)")
    ax.set_ylabel("Zeta Potential (mV)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_conc.png", dpi=300)
    plt.show()

def create_fraction_plots(df):
    """
    Creates:
    - Peak plots (position, width, area) as 2x2 grid (Control + 3 surfactants)
    - Zeta potential plots: absolute and delta vs control
    """
    # Add the condition column based on surfactant concentrations
    df["condition"] = df.apply(surfactant_condition, axis=1)
    df = df.dropna(subset=["condition"])  # remove any rows without a condition

    # Now you can safely use df["condition"]
    conditions = ["Control", "C12E6", "DDAC", "TX100"]

    ...
    
    # Peak properties
    peak_plots = {
        "pos": "Peak Position (nm)",
        "width": "Peak Width (nm)",
        "area": "Peak Area (%)",
    }

    # ----- Peak plots -----
    for key in peak_plots.keys():
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()
    
        for i, cond in enumerate(conditions):
    
            sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
            if sub.empty:
                continue
    
            ax = axes[i]
    
            width = 0.25
    
            for k, (_, row) in enumerate(sub.iterrows()):
    
                peaks = []
                for peak in [1, 2, 3]:
                    peaks.append({
                        "value": row[f"p{peak}_{key}"],
                        "err": row[f"p{peak}_{key}_err"],
                        "area": row[f"p{peak}_area"],
                    })
    
                # order bars by decreasing area
                peaks = sorted(peaks, key=lambda p: p["area"], reverse=True)
    
                for j, p in enumerate(peaks):
    
                    x = k + (j - 1) * width
                    colour = cmap(norm(p["area"]))
    
                    ax.bar(
                        x,
                        p["value"],
                        width=width,
                        yerr=p["err"],
                        capsize=3,
                        color=colour,
                        edgecolor="black",
                    )
    
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([f"{x:.2f}" for x in sub["fraction_DMPG"]])
    
            ax.set_title(cond)
            ax.set_xlabel("DMPG Fraction")
            ax.set_ylabel(peak_labels[key])

        # remove unused subplot if fewer than 4 conditions
        for j in range(len(conditions), len(axes)):
            fig.delaxes(axes[j])
    
        # shared colourbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02)
        cbar.set_label("Peak Area (%)")
    
        fig.savefig(PLOTS_FOLDER / f"{key}_fraction_2x2.png", dpi=300)
        plt.show()

    # ----- Zeta plots (absolute + delta relative to control) -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Absolute zeta
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        if sub.empty:
            continue
        axes[0].errorbar(
            sub["fraction_DMPG"],
            sub["zeta"],
            yerr=sub["zeta_err"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=cond,
            color=colours[cond],
        )

    # Control reference
    control = df[df["condition"] == "Control"][["fraction_DMPG", "zeta", "zeta_err"]].set_index("fraction_DMPG")

    # Control-subtracted zeta
    for cond in conditions[1:]:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        merged = sub.merge(
            control,
            left_on="fraction_DMPG",
            right_index=True,
            suffixes=("_surf", "_ctrl"),
        )
        if merged.empty:
            continue
        delta = merged["zeta_surf"] - merged["zeta_ctrl"]
        delta_err = np.sqrt(merged["zeta_err_surf"]**2 + merged["zeta_err_ctrl"]**2)

        axes[1].errorbar(
            merged["fraction_DMPG"],
            delta,
            yerr=delta_err,
            marker="o",
            linestyle="-",
            capsize=3,
            label=cond,
            color=colours[cond],
        )

    # Reference line at 0
    axes[1].axhline(0, linestyle="--", linewidth=1, color=colours["Control"])

    for ax in axes:
        ax.set_xlabel("Fraction DMPG / (DMPC + DMPG)")
        ax.legend()

    axes[0].set_ylabel("Zeta Potential (mV)")
    axes[1].set_ylabel("Zeta Potential - Control (mV)")

    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_fraction.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plt.close('all')

    folder = DATA_FOLDER / "surfactants" / "50_degrees"

    df = gather_data(folder)

    create_fraction_plots(df)
    
    # folder = DATA_FOLDER / "surfactants" / "25_degrees"

    # df = gather_data(folder)
    
    # create_concentration_plots(df)