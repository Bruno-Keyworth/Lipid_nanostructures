# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:45:00 2026

@author: David
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from itertools import cycle
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER


# ==========================================================
# STYLE
# ==========================================================

def setup_plot_style(n):
    cmap = cmc.hawaii.resampled(n)
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)][::-1]

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    linestyles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['o', 's', 'v', 'D', '^', '*', 'x', 'P'])

    return markers, linestyles


# ==========================================================
# REGEX EXTRACTION
# ==========================================================

def extract_surfactant(name):
    m = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", name)
    return m.group(1) if m else "Unknown"


def extract_microM(name):
    m = re.search(r"(\d+)\s*microM", name)
    return float(m.group(1)) if m else np.nan


def extract_lipid_ratio(name):
    m = re.search(r"(\d+)\s+([A-Z]+)\s*:\s*(\d+)\s+([A-Z]+)", name)
    if m:
        n1, l1, n2, l2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        frac = n2 / (n1 + n2)
        label = f"{n1}:{n2} {l1}:{l2}"
        return frac, label
    return np.nan, "Unknown"


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data(filepath):

    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=[1]
    )

    df.columns = df.columns.str.strip()

    df["Measurement Date and Time"] = pd.to_datetime(
        df["Measurement Date and Time"],
        dayfirst=True,
        errors="coerce"
    )

    df = (
        df.sort_values("Measurement Date and Time")
          .drop_duplicates(subset="Sample Name", keep="last")
    )

    df = df[["Sample Name", "ZP"]].dropna()
    df["ZP"] = df["ZP"].astype(float)

    df["surfactant"] = df["Sample Name"].apply(extract_surfactant)
    df["conc_microM"] = df["Sample Name"].apply(extract_microM)

    ratios = df["Sample Name"].apply(extract_lipid_ratio)
    df["charged_fraction"] = ratios.apply(lambda x: x[0])
    df["ratio_label"] = ratios.apply(lambda x: x[1])

    df = df.dropna(subset=["conc_microM", "charged_fraction", "ZP"])

    return df


# ==========================================================
# STATISTICS
# ==========================================================

def grouped_stats(df, group_cols):

    stats = (
        df.groupby(group_cols)["ZP"]
          .agg(["mean", "std"])
          .reset_index()
    )

    return stats


# ==========================================================
# GENERIC ERRORBAR PLOT
# ==========================================================

def plot_errorbars(stats, x, group_cols, title, xlabel, ylabel, save_name):

    unique_groups = stats[group_cols].drop_duplicates()

    markers, linestyles = setup_plot_style(len(unique_groups))

    plt.figure(figsize=(10, 6))

    for _, row in unique_groups.iterrows():

        mask = np.ones(len(stats), dtype=bool)

        for col in group_cols:
            mask &= stats[col] == row[col]

        sub = stats[mask].sort_values(x)

        ls = next(linestyles)
        mk = next(markers)

        label = " | ".join(str(row[col]) for col in group_cols)

        plt.errorbar(
            sub[x],
            sub["mean"],
            yerr=sub["std"],
            marker=mk,
            linestyle=ls,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=4,
            label=label
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.4)
    plt.legend(bbox_to_anchor=(1.22, 0.5), loc="center")
    plt.tight_layout()

    plt.savefig(PLOTS_FOLDER / save_name, dpi=300)
    plt.show()


# ==========================================================
# FIGURE 1
# ZETA vs SURFACTANT CONCENTRATION
# ==========================================================

def plot_zeta_vs_concentration(df):

    stats = grouped_stats(
        df,
        ["surfactant", "ratio_label", "conc_microM"]
    )

    plot_errorbars(
        stats,
        x="conc_microM",
        group_cols=["surfactant", "ratio_label"],
        title="Zeta potential vs surfactant concentration",
        xlabel="Surfactant concentration (µM)",
        ylabel="Zeta potential (mV)",
        save_name="surfactant_ZETA_vs_concentration.png"
    )


# ==========================================================
# FIGURE 2
# ZETA vs CHARGED FRACTION (FIXED SURFACTANT CONC)
# ==========================================================

def plot_zeta_vs_fraction(df, fixed_conc=100):

    df_fixed = df[df["conc_microM"] == fixed_conc]

    stats = grouped_stats(
        df_fixed,
        ["surfactant", "charged_fraction"]
    )

    plot_errorbars(
        stats,
        x="charged_fraction",
        group_cols=["surfactant"],
        title=f"Zeta vs charged fraction ({fixed_conc} µM)",
        xlabel="Charged lipid fraction",
        ylabel="Zeta potential (mV)",
        save_name="ZETA_vs_charged_fraction.png"
    )

def main():

    filepath = DATA_FOLDER / "surfactant" / "surfactant_zeta.txt"
    df = load_data(filepath)

    plot_zeta_vs_concentration(df)
    plot_zeta_vs_fraction(df, fixed_conc=100)


if __name__ == "__main__":
    main()