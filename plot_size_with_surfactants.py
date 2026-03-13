#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:12:38 2026

@author: brunokeyworth
"""
import json
import numpy as np
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER


# -----------------------------
# Metadata extraction
# -----------------------------

def compute_charged_fraction(name: str):
    match = re.search(r"(\d+)\s*[A-Z]+\s*:\s*(\d+)\s*[A-Z]+", name)
    if match:
        n1, n2 = int(match.group(1)), int(match.group(2))
        return n2 / (n1 + n2)
    return np.nan


def extract_surfactant(name: str):
    match = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", name)
    return match.group(1) if match else "NONE"


def extract_conc_microM(name: str):
    match = re.search(r"(\d+\.?\d*)\s*microM", name)
    return float(match.group(1)) if match else 0.0

def load_averaged_results_from_json(subfolder):

    folder_path = DATA_FOLDER / subfolder
    rows = []

    for file_path in folder_path.glob("*.json"):

        with open(file_path) as f:
            data = json.load(f)

        for entry in data:

            name = entry["base_sample_name"]

            # dominant peak = largest area
            peaks = entry.get("averaged_peaks", [])

            if not peaks:
                continue

            peak = max(peaks, key=lambda p: p.get("mean_area_percent", 0))

            rows.append({
                "sample_name": name,
                "temperature_C": entry.get("temperature_C"),
                "peak_nm": peak.get("mean_peak_position_nm"),
                "sigma_nm": peak.get("mean_peak_width_nm"),
                "charged_fraction": compute_charged_fraction(name),
                "surfactant": extract_surfactant(name),
                "conc_microM": extract_conc_microM(name),
            })

    return pd.DataFrame(rows)



def plot_peak_vs_concentration(df, charged_fraction_filter=0.3):

    df = df[np.isclose(df["charged_fraction"], charged_fraction_filter)]

    groups = df["surfactant"].unique()

    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, surf in enumerate(groups):

        sub = df[df["surfactant"] == surf].sort_values("conc_microM")

        ax[0].plot(
            sub["conc_microM"], sub["peak_nm"],
            marker="o", color=colors[i], label=surf
        )

        ax[1].plot(
            sub["conc_microM"], sub["sigma_nm"],
            marker="o", color=colors[i], label=surf
        )

    for axes in ax:
        axes.set_xlabel("Surfactant concentration (µM)")
        axes.set_title("7 DMPC : 3 DMPG")
        axes.legend()

    ax[0].set_ylabel("Mean peak size (nm)")
    ax[1].set_ylabel("Mean peak width σ (nm)")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "size_vs_surfactant_conc.png", dpi=300)
    plt.show()



def plot_peak_vs_fraction(df, fixed_conc=100):

    df = df[np.isclose(df["conc_microM"], fixed_conc)]

    groups = df["surfactant"].unique()

    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, surf in enumerate(groups):

        sub = df[df["surfactant"] == surf].sort_values("charged_fraction")

        ax[0].plot(
            sub["charged_fraction"], sub["peak_nm"],
            marker="o", color=colors[i], label=surf
        )

        ax[1].plot(
            sub["charged_fraction"], sub["sigma_nm"],
            marker="o", color=colors[i], label=surf
        )

    for axes in ax:
        axes.set_xlabel("Charged lipid fraction")
        axes.set_title(f"Surfactant concentration = {fixed_conc} µM")
        axes.set_yscale("log")
        axes.legend()

    ax[0].set_ylabel("Mean peak size (nm)")
    ax[1].set_ylabel("Mean peak width σ (nm)")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "size_vs_lipid_fraction.png", dpi=300)
    plt.show()

if __name__ == '__main__':

    df = load_averaged_results_from_json("surfactants/averaged_peaks")
    
    plot_peak_vs_concentration(df)
    plot_peak_vs_fraction(df)