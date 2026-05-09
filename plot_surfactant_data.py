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
    "font.size": 16,          # base size
    "axes.titlesize": 16,     # subplot titles
    "axes.labelsize": 16,     # x and y labels
    "xtick.labelsize": 15,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
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

label_map = {
    "Control": "Control",
    "C12E6": r"$C_{12}E_6$",
    "DDAC": "DDAC",
    "TX100": "Triton X-100",
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

def plot_peak_bars(ax, rows, key, width=0.25):
    """
    Plot bars for peaks on a given axis. Positions are integer indices; x-axis labels are separate.
    """
    for k, (_, row) in enumerate(rows.iterrows()):
        peaks = [
            {"value": row[f"p{i}_{key}"], "err": row[f"p{i}_{key}_err"], "area": row[f"p{i}_area"]}
            for i in [1, 2, 3]
        ]
        peaks = sorted(peaks, key=lambda p: p["area"], reverse=True)
        for j, p in enumerate(peaks):
            x = k + (j - 1) * width  # keep original indexing
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
            
def rows_label_values(rows):
    # For concentration plots, take the nonzero surfactant value + zeros for control
    surfactants = ["C12E6", "DDAC", "TX100"]
    if all(col in rows.columns for col in surfactants):
        # concentration plot: find which surfactant is nonzero in each row
        vals = []
        for _, row in rows.iterrows():
            nz = [row[surf] for surf in surfactants if row[surf] > 0]
            vals.append(nz[0] if nz else 0)
        return vals
    # otherwise, assume fraction plot and take "fraction_DMPG"
    if "fraction_DMPG" in rows.columns:
        return rows["fraction_DMPG"].values
    return np.arange(len(rows))

def create_peak_figure(df_rows, titles, xlabel_vals=None,
                       xlabel_name=None, fig_shape=(2, 3),
                       filename_prefix="plot"):

    fig, axes = plt.subplots(
        fig_shape[0], fig_shape[1],
        figsize=(6*fig_shape[1], 6*fig_shape[0]),
        constrained_layout=True,
        sharex='col',
        sharey='row'
    )

    axes = np.array(axes)

    row_keys = ["pos", "width"]

    for col, (rows, title) in enumerate(zip(df_rows, titles)):

        if rows.empty:
            continue

        for row, key in enumerate(row_keys):
            ax = axes[row, col]

            x_pos = np.arange(len(rows))

            plot_peak_bars(ax, rows, key, width=0.25)

            # x-axis only on bottom row
            if row == 1:
                if xlabel_vals is not None:
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f"{v:g}" for v in xlabel_vals[col]])
                else:
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([str(v) for v in x_pos])
            else:
                ax.set_xticks([])

            # if row == 0:
            #     ax.set_title(label_map.get(title, title), fontsize=26)

            ax.margins(y=0.05)

    axes[0, 0].set_ylabel("Peak Position (nm)", fontsize=26)
    axes[1, 0].set_ylabel("Peak Width (nm)", fontsize=26)
    for ax in axes[0, :]:
        ax.set_yscale('log')
    for ax in axes[1, :]:
        ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.relim()
        ax.autoscale_view()
        ax.tick_params(labelsize=16)
    sub = [r'\textbf{(a)} Control', r'$\textbf{(b)}\ C_{12}E_6$', r'\textbf{(c)} DDAC', r'\textbf{(d)} Triton X-100', 
           r'\textbf{(e)}', r'\textbf{(f)}',
           r'\textbf{(g)}', r'\textbf{(h)}']
    
    for i, ax in enumerate(axes.flat):
        ax.text(
            0.02, 0.97, sub[i],
            transform=ax.transAxes,
            fontsize=20,
            fontweight='bold',
            va='top',
            ha='left'
        )
    fig.supxlabel(xlabel_name, fontsize=28)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("Relative Peak Intensity (\%)", fontsize=26)

    fig.savefig(PLOTS_FOLDER / f"{filename_prefix}.png", dpi=300)
    plt.show()

# --- Refactored concentration plot ---
def create_concentration_plots(df, ratio=0.3, zeta=True):
    df = df[np.isclose(df["fraction_DMPG"], ratio)]
    surfactants = ["C12E6", "DDAC", "TX100"]
    control = df[(df[surfactants] == 0).all(axis=1)]
    
    df_rows = []
    x_vals = []
    titles = []
    for surf in surfactants:
        sub = df[df[surf] > 0].sort_values(surf)
        if sub.empty:
            df_rows.append(pd.DataFrame())
            x_vals.append([])
            titles.append(surf)
            continue
        rows = pd.concat([control, sub], ignore_index=True)
        df_rows.append(rows)
        x_vals.append(np.concatenate([np.full(len(control), 0), sub[surf].values]))
        titles.append(surf)
    
    create_peak_figure(
        df_rows=df_rows,
        titles=titles,
        xlabel_vals=x_vals,
        xlabel_name=r"Surfactant concentration ($\mu$M)",
        fig_shape=(2, 3),
        filename_prefix="concentration"
    )
    # if zeta:
    #     plot_zeta_against_concentration(df, surfactants)

# --- Refactored fraction plot ---
def create_fraction_plots(df, zeta=True):
    df["condition"] = df.apply(surfactant_condition, axis=1)
    df = df.dropna(subset=["condition"])
    conditions = ["Control", "C12E6", "DDAC", "TX100"]
    
    df_rows = []
    x_vals = []
    titles = []
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        df_rows.append(sub)
        x_vals.append(sub["fraction_DMPG"].values if not sub.empty else [])
        titles.append(cond)
    
    create_peak_figure(
        df_rows=df_rows,
        titles=titles,
        xlabel_vals=x_vals,
        xlabel_name="DMPG fraction",
        fig_shape=(2, 4),
        filename_prefix="fraction"
    )
    # if zeta:
    #     plot_zeta_against_fraction(df, conditions)

def plot_zeta_against_concentration(df, surfactants):
    # Identify control (no surfactant added)
    control = df[(df["C12E6"] == 0) & (df["DDAC"] == 0) & (df["TX100"] == 0)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if not control.empty:
        control_zeta = control["zeta"].mean()
        control_err = control["zeta_err"].mean()
        ax.axhline(
            control_zeta,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Control"
        )
        ax.fill_between(
            [0, max(df[surfactants].max())],
            control_zeta - control_err,
            control_zeta + control_err,
            color="black",
            alpha=0.1
        )
    for surf in surfactants:
        sub = df[df[surf] > 0].sort_values(surf)
        if sub.empty:
            continue

        x = sub[surf].values
        y = sub["zeta"].values
        yerr = sub["zeta_err"].values

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label_map.get(surf, surf),
            color=colours[surf]
        )
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Surfactant Concentration (µM)", fontsize=22)
    ax.set_ylabel(r"$\zeta$ (mV)", fontsize=22)
    ax.set_xscale('log')
    ax.legend(fontsize=22, framealpha=0)

    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_conc.png", dpi=300)
    plt.show()

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
            label=label_map.get(cond, cond)
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
            label=label_map.get(cond, cond)
        )
    
    # Reference line at 0
    axes[1].axhline(0, linestyle="--", linewidth=1, color=colours["Control"])
    
    # Axis labels
    axes[0].set_ylabel(r"$\zeta$ (mV)", fontsize=22)
    axes[1].set_ylabel(r"$\Delta\zeta$ (mV)", fontsize=22)
    for label, ax in {r'\textbf{(a)}': axes[0], 
                      r'\textbf{(b)}': axes[1]}.items(): 
        ax.set_xlabel("DMPG Fraction", fontsize=22)
        #ax.tick_params(labelsize=16)
        ax.text(
                0.02, 0.97, label,
                transform=ax.transAxes,
                fontsize=20,
                fontweight='bold',
                va='top', ha='left'
            )
    
    # Create common legend without duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # deduplicate
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=len(by_label),
        frameon=False,
        fontsize=22,
        bbox_to_anchor=(0.5, -0.15)
    )
    
    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_fraction.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plt.close('all')

    folder = DATA_FOLDER / "surfactants" / "50_degrees"
    df = gather_data(folder)
    create_fraction_plots(df)
    
    # folder = DATA_FOLDER / "surfactants" / "25_degrees"
    # df = gather_data(folder)
    # create_concentration_plots(df)