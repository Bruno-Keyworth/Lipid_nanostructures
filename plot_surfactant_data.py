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
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
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
                zorder=3
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

    import matplotlib.gridspec as gridspec

    ncols = fig_shape[1]

    fig = plt.figure(
        figsize=(6*ncols, 10),
        constrained_layout=True
    )

    # 3 rows:
    # 0 = peak position
    # 1 = upper broken width axis
    # 2 = lower broken width axis
    gs = gridspec.GridSpec(
        3, ncols,
        height_ratios=[2, 0.7, 1.5],
        hspace=0.01,
        figure=fig
    )

    axes_pos = []
    axes_width_top = []
    axes_width_bottom = []

    for col in range(ncols):

        ax_pos = fig.add_subplot(
            gs[0, col],
            sharey=axes_pos[0] if axes_pos else None
        )
        axes_pos.append(ax_pos)

        ax_w_top = fig.add_subplot(
            gs[1, col],
            sharey=axes_width_top[0] if axes_width_top else None
        )
        axes_width_top.append(ax_w_top)

        ax_w_bot = fig.add_subplot(
            gs[2, col],
            sharex=ax_w_top,
            sharey=axes_width_bottom[0] if axes_width_bottom else None
        )
        axes_width_bottom.append(ax_w_bot)

    row_keys = ["pos", "width"]
    all_axes = np.array([axes_pos, axes_width_top, axes_width_bottom]).reshape(3, ncols)

    for r in range(all_axes.shape[0]):
        for c in range(all_axes.shape[1]):
    
            ax = all_axes[r, c]
    
            if c == 0:
                ax.tick_params(
                    axis='y',
                    left=True,
                    labelleft=True
                )
            else:
                ax.tick_params(
                    axis='y',
                    left=False,
                    labelleft=False
                )
                ax.tick_params(labelleft=False)
    for col, (rows, title) in enumerate(zip(df_rows, titles)):

        if rows.empty:
            continue

        x_pos = np.arange(len(rows))

        # -----------------------------
        # Peak position row
        # -----------------------------
        ax = axes_pos[col]

        ax.grid(zorder=0)

        plot_peak_bars(ax, rows, "pos", width=0.25)

        ax.set_yscale('log')
        ax.set_yticks([
            10**2, 2*10**2, 3*10**2, 4*10**2,
            5*10**2, 6*10**2, 7*10**2,
            8*10**2, 9*10**2, 10**3
        ])

        ax.set_xticks([])

        ax.margins(y=0.05)

        # -----------------------------
        # Peak width broken axes
        # -----------------------------
        ax_top = axes_width_top[col]
        ax_bot = axes_width_bottom[col]

        ax_top.grid(zorder=0)
        ax_bot.grid(zorder=0)

        # Plot onto BOTH axes
        plot_peak_bars(ax_top, rows, "width", width=0.25)
        plot_peak_bars(ax_bot, rows, "width", width=0.25)

        # Broken ranges
        ax_bot.set_ylim(0, 250)
        ax_top.set_ylim(250, 1500)

        # Hide touching spines
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)

        ax_top.tick_params(labelbottom=False)

        # Break marks
        d = 0.015

        kwargs = dict(
            transform=ax_top.transAxes,
            color='k',
            clip_on=False
        )

        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)

        kwargs.update(transform=ax_bot.transAxes)

        ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        # X labels only on lower width row
        if xlabel_vals is not None:
            ax_bot.set_xticks(x_pos)
            ax_bot.set_xticklabels(
                [f"{v:g}" for v in xlabel_vals[col]]
            )
        else:
            ax_bot.set_xticks(x_pos)
            ax_bot.set_xticklabels([str(v) for v in x_pos])

        ax_top.margins(y=0.05)
        ax_bot.margins(y=0.05)

    # Labels
    axes_pos[0].set_ylabel("Peak Position (nm)", fontsize=30)
    axes_width_bottom[0].set_ylabel("Peak Width (nm)", fontsize=30)

    # Tick styling
    for ax in axes_pos + axes_width_top + axes_width_bottom:
        ax.relim()
        ax.autoscale_view()
        ax.tick_params(labelsize=22)

    sub = [
        r'\textbf{(a)} Control',
        r'$\textbf{(b)}\ C_{12}E_6$',
        r'\textbf{(c)} DDAC',
        r'\textbf{(d)} Triton X-100',
        r'\textbf{(e)}',
        r'\textbf{(f)}',
        r'\textbf{(g)}',
        r'\textbf{(h)}'
    ]

    if xlabel_name == r"Surfactant concentration ($\mu$M)":
        sub = [
            r'$\textbf{(a)}\ C_{12}E_6$',
            r'\textbf{(b)} DDAC',
            r'\textbf{(c)} Triton X-100',
            r'\textbf{(d)}',
            r'\textbf{(e)}',
            r'\textbf{(f)}'
        ]

    all_axes_for_labels = (
        axes_pos +
        axes_width_top +
        axes_width_bottom
    )

    for i, ax in enumerate(all_axes_for_labels[:len(sub)]):
        ax.text(
            0.02, 0.97, sub[i],
            transform=ax.transAxes,
            fontsize=26,
            fontweight='bold',
            va='top',
            ha='left',
            zorder=3
        )

    fig.supxlabel(xlabel_name, fontsize=28)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=all_axes_for_labels,
        pad=0.02
    )

    cbar.set_label(
        "Relative Peak Intensity (\%)",
        fontsize=30
    )

    fig.savefig(
        PLOTS_FOLDER / f"{filename_prefix}.png",
        dpi=300
    )

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

    # folder = DATA_FOLDER / "surfactants" / "50_degrees"
    # df = gather_data(folder)
    # create_fraction_plots(df)
    
    folder = DATA_FOLDER / "surfactants" / "25_degrees"
    df = gather_data(folder)
    create_concentration_plots(df)