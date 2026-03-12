# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:45:00 2026

@author: David
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
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
        if frac > 0:
            label = f"{l1}:{l2}"
        else: 
            label = l1
        return frac, label
    return np.nan, "Unknown"


# ==========================================================
# DATA LOADING (JSON)
# ==========================================================

def load_surfactant_json(folder: Path) -> pd.DataFrame:
    """
    Loads all JSON files in a surfactants folder.
    Expects each JSON file to be a list of dicts with at least:
        - 'sample_name'
        - 'zeta_mV'
    """
    rows = []
    for file_path in folder.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        for entry in data:
            if entry.get("type") == "zeta":  # Only keep Zeta measurements
                rows.append({
                    "Sample Name": entry.get("sample_name"),
                    "ZP": entry.get("zeta_mV")
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No Zeta data found in {folder}")
    df["ZP"] = df["ZP"].astype(float)
    return df


# ==========================================================
# DATA CLEANING
# ==========================================================

def clean_zeta_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df["surfactant"] = df["Sample Name"].apply(extract_surfactant)
    df["conc_microM"] = df["Sample Name"].apply(extract_microM)

    ratios = df["Sample Name"].apply(extract_lipid_ratio)
    df["charged_fraction"] = ratios.apply(lambda x: x[0])
    df["lipid_label"] = ratios.apply(lambda x: x[1])

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

def build_style_maps(df):

    surfactants = sorted(df["surfactant"].unique())
    lipids = sorted(df["lipid_label"].unique())

    # Colours per surfactant
    cmap = cmc.hawaii.resampled(len(surfactants))
    surfactant_colors = {
        s: mcolors.to_hex(cmap(i))
        for i, s in enumerate(surfactants)
    }

    # Marker/linestyle per lipid
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '--', '-.', ':']

    lipid_styles = {
        l: (markers[i % len(markers)], linestyles[i % len(linestyles)])
        for i, l in enumerate(lipids)
    }

    return surfactant_colors, lipid_styles


def plot_errorbars(stats, x, group_cols, title, xlabel, ylabel, save_name):

    unique_groups = stats[group_cols].drop_duplicates()
    surfactant_colours, lipid_styles = build_style_maps(stats)
    plt.figure(figsize=(10, 6))

    for _, row in unique_groups.iterrows():

        mask = np.ones(len(stats), dtype=bool)
        for col in group_cols:
            mask &= stats[col] == row[col]

        sub = stats[mask].sort_values(x)

        surf = row["surfactant"]
        lipid = row["lipid_label"]
        
        colour = surfactant_colours[surf]
        mk, ls = lipid_styles[lipid]

        label = " | ".join(str(row[col]) for col in group_cols)

        plt.errorbar(
            sub[x],
            sub["mean"],
            yerr=sub["std"],
            marker=mk,
            linestyle=ls,
            color=colour,
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

def remove_single_concentration_groups(df):

    conc_counts = (
        df.groupby(["surfactant", "lipid_label"])["conc_microM"]
        .nunique()
        .reset_index(name="n_conc")
    )

    valid = conc_counts[conc_counts["n_conc"] > 1][["surfactant", "lipid_label"]]

    df_filtered = df.merge(valid, on=["surfactant", "lipid_label"])

    return df_filtered

# ==========================================================
# FIGURES
# ==========================================================

def plot_zeta_vs_concentration(df):

    allowed_ratios = {
        "DMPC:DMPG",
        "POPC"
    }

    df = df[df["lipid_label"].isin(allowed_ratios)]

    # Separate baseline (no surfactant)
    df_none = df[df["surfactant"] == "NONE"]
    df_real = df[df["surfactant"] != "NONE"]

    augmented_rows = [df_real]

    for ratio in df["lipid_label"].unique():

        base = df_none[df_none["lipid_label"] == ratio]

        surfactants = df_real[df_real["lipid_label"] == ratio]["surfactant"].unique()

        for s in surfactants:
            temp = base.copy()
            temp["surfactant"] = s
            augmented_rows.append(temp)

    df_augmented = pd.concat(augmented_rows, ignore_index=True)
    df_augmented = remove_single_concentration_groups(df_augmented)

    stats = grouped_stats(df_augmented, ["surfactant", "lipid_label", "conc_microM"])

    plot_errorbars(
        stats,
        x="conc_microM",
        group_cols=["surfactant", "lipid_label"],
        title="Zeta potential vs surfactant concentration",
        xlabel="Surfactant concentration (µM)",
        ylabel="Zeta potential (mV)",
        save_name="surfactant_ZETA_vs_concentration.png"
    )



def plot_zeta_vs_fraction(df, fixed_conc=100):
    df_fixed = df[np.isclose(df["conc_microM"], fixed_conc) | np.isclose(df["conc_microM"], 0)]
    stats = grouped_stats(df_fixed, ["surfactant", "charged_fraction"])
    plot_errorbars(
        stats,
        x="charged_fraction",
        group_cols=["surfactant"],
        title=f"Zeta vs charged fraction ({fixed_conc} µM)",
        xlabel="Charged lipid fraction",
        ylabel="Zeta potential (mV)",
        save_name="ZETA_vs_charged_fraction.png"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    surf_folder = DATA_FOLDER / "surfactants"
    df = load_surfactant_json(surf_folder)
    df_clean = clean_zeta_dataframe(df)

    plot_zeta_vs_concentration(df_clean)
    plot_zeta_vs_fraction(df_clean, fixed_conc=100)


if __name__ == "__main__":
    main()