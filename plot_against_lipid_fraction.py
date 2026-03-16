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
from extract_peak_data import gather_data

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

def create_concentration_plots(df, ratio=0.3, zeta=True):
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
        
    if zeta:
        plot_zeta_against_concentration(df, surfactants)
        
        
def plot_zeta_against_concentration(df, surfactants):
    
    control = df[(df["C12E6"] == 0) & (df["DDAC"] == 0) & (df["TX100"] == 0)]

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

def create_fraction_plots(df, zeta=True):
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
    if zeta:   
        plot_zeta_against_fraction(df, conditions)
        
def plot_zeta_against_fraction(df, conditions):

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
            color=colours[cond],
            label=cond  # only needed for legend
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
            color=colours[cond],
            label=cond  # only needed for legend
        )
    
    # Reference line at 0
    axes[1].axhline(0, linestyle="--", linewidth=1, color=colours["Control"])
    
    # Axis labels
    axes[0].set_ylabel("Zeta Potential (mV)")
    axes[1].set_ylabel("Zeta Potential - Control (mV)")
    for ax in axes:
        ax.set_xlabel("DMPG Fraction")
    
    # Create common legend without duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # deduplicate
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=len(by_label),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.1)
    )
    
    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_fraction.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plt.close('all')

    folder = DATA_FOLDER / "surfactants" / "50_degrees"
    df = gather_data(folder)
    create_fraction_plots(df)
    
    folder = DATA_FOLDER / "surfactants" / "25_degrees"
    df = gather_data(folder)
    create_concentration_plots(df)