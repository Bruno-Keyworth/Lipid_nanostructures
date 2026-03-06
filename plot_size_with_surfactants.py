#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:12:38 2026

@author: brunokeyworth
"""

import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
from io import StringIO

from get_filepaths import DATA_FOLDER
from get_standard_deviation import fit_gaussian

# -------------------------------------------------
# Regex metadata extraction
# -------------------------------------------------

def extract_metadata(name):

    lipid_match = re.search(r"(\d+)\s+([A-Z0-9]+)\s*:\s*(\d+)\s+([A-Z0-9]+)", name)
    surf_match = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", name)
    conc_match = re.search(r"(\d+\.?\d*)\s*microM", name)

    if lipid_match:
        n1, l1, n2, l2 = lipid_match.groups()
        n1, n2 = int(n1), int(n2)
        frac = n2 / (n1 + n2)
    else:
        l1 = l2 = None
        frac = np.nan

    surfactant = surf_match.group(1) if surf_match else None
    conc = float(conc_match.group(1)) if conc_match else np.nan

    return {
        "lipid1": l1,
        "lipid2": l2,
        "charged_fraction": frac,
        "surfactant": surfactant,
        "surfactant_conc_microM": conc
    }

def process_dls_surfactant(csv_path, encoding="latin1", sep="\t"):
    with open(DATA_FOLDER / csv_path, "r", encoding=encoding) as f:
        text = f.read()

    # Fix missing space between surfactant name and concentration
    # e.g. 'C12E6100' -> 'C12E6 100'
    text = re.sub(r'(C\d+E\d+)(\d+)', r'\1 \2', text)

    # Load into DataFrame
    df = pd.read_csv(
        StringIO(text),
        sep=sep,
        engine="python",
        on_bad_lines="skip"
    )

    size_cols = [c for c in df.columns if c.startswith("Sizes[")]
    intensity_cols = [c for c in df.columns if c.startswith("Intensities[")]
    volume_cols = [c for c in df.columns if c.startswith("Volumes[")]

    corr_cols = [c for c in df.columns if c.startswith("Correlation Data[")]
    delay_cols = [c for c in df.columns if c.startswith("Correlation Delay Times[")]

    scalar_cols = [
        c for c in df.columns
        if c not in size_cols + intensity_cols + volume_cols + corr_cols + delay_cols
    ]

    df_valid = df[df["Sample Name"] != "Sample Name"]

    df_valid = df_valid.drop_duplicates(
        subset=["Sample Name", "Measurement Date and Time"]
    )

    data = []

    for _, row in df_valid.iterrows():

        name = row["Sample Name"]

        meta = row[scalar_cols].to_dict()
        parsed = extract_metadata(name)

        size_row = row[size_cols].to_numpy()
        intensity_row = row[intensity_cols].to_numpy()
        volume_row = row[volume_cols].to_numpy()

        corr_row = row[corr_cols].to_numpy()
        delay_row = row[delay_cols].to_numpy()

        data.append({
            "meta": meta,
            "parsed": parsed,
            "size": np.column_stack(
                (size_row, intensity_row, volume_row)
            ).T.tolist(),
            "correlation": np.column_stack(
                (corr_row, delay_row)
            ).T.tolist(),
        })

    outfile = DATA_FOLDER / "surfactant" / "surfactant_dls.json"

    os.makedirs(outfile.parent, exist_ok=True)

    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)


def load_surfactant_results():

    file = DATA_FOLDER / "surfactant" / "surfactant_dls.json"

    with open(file) as f:
        data = json.load(f)

    rows = []

    for m in data:

        meta = m["meta"]
        parsed = m["parsed"]

        try:
            peak = float(meta["CONTIN Peaks[1]"])
        except:
            continue

        try:
            sizes = np.asarray(m["size"][0], float)
            intensities = np.asarray(m["size"][1], float)

            stacked = np.column_stack((sizes, intensities))
            sigma = fit_gaussian(stacked, PLOT=False)

        except:
            sigma = np.nan

        rows.append({
            "peak_nm": peak,
            "sigma_nm": sigma,
            "surfactant": parsed["surfactant"],
            "conc_microM": parsed["surfactant_conc_microM"],
            "charged_fraction": parsed["charged_fraction"]
        })

    return pd.DataFrame(rows)

def plot_peak_vs_concentration(df):
    stats = (
        df.groupby(["surfactant", "charged_fraction", "conc_microM"])
          .agg(
              mean_peak=("peak_nm", "mean"),
              std_peak=("peak_nm", "std"),
              mean_sigma=("sigma_nm", "mean"),
              std_sigma=("sigma_nm", "std")
          )
          .reset_index()
    )

    groups = stats[["surfactant", "charged_fraction"]].drop_duplicates()
    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    # Peak size
    plt.figure()
    for i, (_, g) in enumerate(groups.iterrows()):
        sub = stats[
            (stats["surfactant"] == g["surfactant"]) &
            (stats["charged_fraction"] == g["charged_fraction"])
        ].sort_values("conc_microM")
        label = f"{g['surfactant']} | f={g['charged_fraction']:.2f}"
        plt.errorbar(
            sub["conc_microM"],
            sub["mean_peak"],
            yerr=sub["std_peak"],
            marker="o",
            color=colors[i],
            label=label
        )
    plt.xlabel("Surfactant concentration (µM)")
    plt.ylabel("Peak size (nm)")
    plt.title("Peak size vs surfactant concentration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Peak width
    plt.figure()
    for i, (_, g) in enumerate(groups.iterrows()):
        sub = stats[
            (stats["surfactant"] == g["surfactant"]) &
            (stats["charged_fraction"] == g["charged_fraction"])
        ].sort_values("conc_microM")
        label = f"{g['surfactant']} | f={g['charged_fraction']:.2f}"
        plt.errorbar(
            sub["conc_microM"],
            sub["mean_sigma"],
            yerr=sub["std_sigma"],
            marker="o",
            color=colors[i],
            label=label
        )
    plt.xlabel("Surfactant concentration (µM)")
    plt.ylabel("Peak width σ (nm)")
    plt.title("Peak width vs surfactant concentration")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_peak_vs_fraction(df, fixed_conc=100):
    df_fixed = df[df["conc_microM"] == fixed_conc]

    stats = (
        df_fixed.groupby(["surfactant", "charged_fraction"])
                .agg(
                    mean_peak=("peak_nm", "mean"),
                    std_peak=("peak_nm", "std"),
                    mean_sigma=("sigma_nm", "mean"),
                    std_sigma=("sigma_nm", "std")
                )
                .reset_index()
    )

    unique_surfactants = stats["surfactant"].unique()
    cmap = cmc.hawaii.resampled(len(unique_surfactants))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    # Peak size vs fraction
    plt.figure()
    for i, surf in enumerate(unique_surfactants):
        sub = stats[stats["surfactant"] == surf].sort_values("charged_fraction")
        plt.errorbar(
            sub["charged_fraction"],
            sub["mean_peak"],
            yerr=sub["std_peak"],
            marker="o",
            color=colors[i],
            label=surf
        )
    plt.xlabel("Charged lipid fraction")
    plt.ylabel("Peak size (nm)")
    plt.title(f"Peak size vs charged fraction (conc={fixed_conc} µM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Peak width vs fraction
    plt.figure()
    for i, surf in enumerate(unique_surfactants):
        sub = stats[stats["surfactant"] == surf].sort_values("charged_fraction")
        plt.errorbar(
            sub["charged_fraction"],
            sub["mean_sigma"],
            yerr=sub["std_sigma"],
            marker="o",
            color=colors[i],
            label=surf
        )
    plt.xlabel("Charged lipid fraction")
    plt.ylabel("Peak width σ (nm)")
    plt.title(f"Peak width vs charged fraction (conc={fixed_conc} µM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

process_dls_surfactant("surfactants.csv")
df = load_surfactant_results()

plot_peak_vs_concentration(df)
plot_peak_vs_fraction(df)
