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

def create_peak_figure(df_rows, keys, titles, xlabel_vals=None, xlabel_name=None, fig_shape=(1, 3), filename_prefix="plot"):
    """
    Create peak plots with integer bar positions but custom x-axis labels.
    
    Parameters
    ----------
    df_rows : list of pd.DataFrame
        One DataFrame per subplot/condition.
    keys : list of str
        'pos', 'width', 'area'.
    titles : list of str
        Titles for each subplot.
    xlabel_vals : list of lists, optional
        List of numeric values for x-axis labels (one list per subplot). If None, uses indices.
    xlabel_name : str, optional
        Name to show below x-axis (e.g. 'Lipid fraction', 'Surfactant concentration')
    fig_shape : tuple
        Grid shape.
    filename_prefix : str
        Prefix for saved figure.
    """
    for key in keys:
        fig, axes = plt.subplots(*fig_shape, figsize=(6*fig_shape[1], 4*fig_shape[0]), constrained_layout=True)
        axes = np.array(axes).flatten()
        
        for i, (ax, rows, title) in enumerate(zip(axes, df_rows, titles)):
            if rows.empty:
                continue
            
            # Inside create_peak_figure, for each subplot:
            x_pos = np.arange(len(rows))  # integer bar positions
            plot_peak_bars(ax, rows, key, width=0.25)  # positions handled inside plot_peak_bars
            
            # Set ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{v:g}" for v in xlabel_vals[i]])  # numeric values for display
            if xlabel_name:
                ax.set_xlabel(xlabel_name)
            
            # tick labels
            if xlabel_vals is not None:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{v:g}" for v in xlabel_vals[i]])
            else:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(v) for v in x_pos])
            
            ax.set_title(title)
            ax.set_ylabel(peak_labels[key])
            if key == 'pos':
                ax.set_yscale('log')
            else: 
                ax.set_ylim(bottom=0)
            ax.margins(y=0.05)
            if xlabel_name:
                ax.set_xlabel(xlabel_name)
        
        for ax in axes[len(df_rows):]:
            fig.delaxes(ax)
        
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02)
        cbar.set_label("Peak Area (%)")
        
        fig.savefig(PLOTS_FOLDER / f"{key}_{filename_prefix}.png", dpi=300)
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
    keys=["pos", "width", "area"],
    titles=titles,
    xlabel_vals=x_vals,              # surfactant concentrations
    xlabel_name=r"Surfactant concentration ($\mu$M)",
    fig_shape=(1,3),
    filename_prefix="concentration"
    )
    if zeta:
        plot_zeta_against_concentration(df, surfactants)

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
    keys=["pos", "width", "area"],
    titles=titles,
    xlabel_vals=x_vals,              # DMPG fractions
    xlabel_name="DMPG fraction",
    fig_shape=(2,2),
    filename_prefix="fraction"
    )
    if zeta:
        plot_zeta_against_fraction(df, conditions)

def plot_zeta_against_concentration(df, surfactants):
    # Identify control (no surfactant added)
    control = df[(df["C12E6"] == 0) & (df["DDAC"] == 0) & (df["TX100"] == 0)]

    fig, ax = plt.subplots()

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
            label=surf,
            color=colours[surf]
        )

    ax.set_xlabel("Surfactant concentration (µM)")
    ax.set_ylabel("Zeta Potential (mV)")
    ax.set_xscale('log')
    ax.legend()

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