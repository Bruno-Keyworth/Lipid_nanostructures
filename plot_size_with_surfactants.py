#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:12:38 2026

@author: brunokeyworth
"""
import os
import json
import numpy as np
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER
from get_standard_deviation import fit_gaussian

def extract_metadata(name):
    # Match lipid counts
    lipid_match = re.search(r"(\d+)\s*([A-Z0-9]+)\s*:\s*(\d+)\s*([A-Z0-9]+)", name)
    if lipid_match:
        n1, l1, n2, l2 = lipid_match.groups()
        n1, n2 = int(n1), int(n2)
        frac = n2 / (n1 + n2) if (n1 + n2) > 0 else 0
    else:
        l1 = l2 = None
        frac = np.nan

    # Match surfactant
    surf_match = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", name)
    surfactant = surf_match.group(1) if surf_match else None

    # Match surfactant concentration
    conc_match = re.search(r"(\d+\.?\d*)\s*microM", name)
    conc = float(conc_match.group(1)) if conc_match else np.nan

    return {
        "lipid1": l1,
        "lipid2": l2,
        "charged_fraction": frac,
        "surfactant": surfactant,
        "surfactant_conc_microM": conc
    }

# --- Loading function ---
def load_surfactant_results_from_json(subfolder: str) -> pd.DataFrame:
    """
    Load all JSON files from DATA_FOLDER / subfolder.
    """
    folder_path = DATA_FOLDER / subfolder
    rows = []

    for file_path in folder_path.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        for entry in data:
            entry["filename"] = file_path.name
            rows.append(entry)

    df = pd.DataFrame(rows)
    if "sample_name" not in df.columns:
        raise ValueError("Your JSON files must contain 'sample_name' field")

    # Compute charged_fraction
    def compute_charged_fraction(name: str) -> float:
        match = re.search(r"(\d+)\s*[A-Z]+\s*:\s*(\d+)\s*[A-Z]+", name)
        if match:
            n1, n2 = int(match.group(1)), int(match.group(2))
            return n2 / (n1 + n2)
        else:
            return np.nan

    df["charged_fraction"] = df["sample_name"].apply(compute_charged_fraction)

    # Extract surfactant
    def extract_surfactant(name: str) -> str:
        match = re.search(r"mg_ml\s*([A-Za-z0-9]+)", name)
        return match.group(1) if match else "NONE"

    # Extract concentration
    def extract_conc_microM(name: str) -> float:
        match = re.search(r"(\d+)\s*microM", name)
        return float(match.group(1)) if match else 0.0

    df["surfactant"] = df["sample_name"].apply(extract_surfactant)
    df["conc_microM"] = df["sample_name"].apply(extract_conc_microM)

    return df

def largest_area_peak(peaks):
    if not peaks:
        return pd.Series({"peak_nm": np.nan, "sigma_nm": np.nan})
    peak = max(peaks, key=lambda p: p.get("area_percent", 0))
    return pd.Series({"peak_nm": peak.get("mean_nm", np.nan),
                      "sigma_nm": peak.get("size_peak_nm", np.nan)})

def plot_peak_vs_concentration(df, charged_fraction_filter=0.3):
    # Only use size measurements
    df_size = df[df["type"] == "size"]

    # Filter for desired charged fraction
    df_size = df_size[df_size["charged_fraction"] == charged_fraction_filter]

    # Extract the peak with largest area

    df_size[["peak_nm", "sigma_nm"]] = df_size["peaks"].apply(largest_area_peak)

    stats = (
        df_size.groupby(["surfactant", "conc_microM"])
               .agg(mean_peak=("peak_nm", "mean"),
                    std_peak=("peak_nm", "std"),
                    mean_sigma=("sigma_nm", "mean"),
                    std_sigma=("sigma_nm", "std"))
               .reset_index()
    )


    groups = stats["surfactant"].unique()
    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, surf in enumerate(groups):
        sub = stats[stats["surfactant"] == surf].sort_values("conc_microM")
        ax[0].errorbar(sub["conc_microM"], sub["mean_peak"], yerr=sub["std_peak"], marker="o", color=colors[i], label=surf)
        ax[1].errorbar(sub["conc_microM"], sub["mean_sigma"], yerr=sub["std_sigma"], marker="o", color=colors[i], label=surf)

    for axes in ax:
        axes.set_xlabel("Surfactant concentration (µM)")
        axes.set_xscale('log')
        axes.set_title("7 DMPC : 3 DMPG")
        axes.legend()
    ax[0].set_ylabel("Peak size (nm)")
    ax[1].set_ylabel("Peak width σ (nm)")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / 'size_vs_surfactant_conc.png', dpi=300)
    plt.show()

def plot_peak_vs_fraction(df, fixed_conc=100):
    df_size = df[df["type"] == "size"]
    df_fixed = df_size[df_size["conc_microM"] == fixed_conc]

    df_fixed[["peak_nm", "sigma_nm"]] = df_fixed["peaks"].apply(largest_area_peak)

    stats = (
        df_fixed.groupby(["surfactant", "charged_fraction"])
                .agg(mean_peak=("peak_nm", "mean"),
                     std_peak=("peak_nm", "std"),
                     mean_sigma=("sigma_nm", "mean"),
                     std_sigma=("sigma_nm", "std"))
                .reset_index()
    )

    unique_surfactants = stats["surfactant"].unique()
    cmap = cmc.hawaii.resampled(len(unique_surfactants))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, surf in enumerate(unique_surfactants):
        sub = stats[stats["surfactant"] == surf].sort_values("charged_fraction")
        ax[0].errorbar(sub["charged_fraction"], sub["mean_peak"], yerr=sub["std_peak"], marker="o", color=colors[i], label=surf)
        ax[1].errorbar(sub["charged_fraction"], sub["mean_sigma"], yerr=sub["std_sigma"], marker="o", color=colors[i], label=surf)

    for axes in ax:
        axes.set_xlabel("Charged lipid fraction")
        axes.set_title(f"Surfactant Concentration = {fixed_conc} µM")
        axes.set_yscale('log')
        axes.legend()
    ax[0].set_ylabel("Peak size (nm)")
    ax[1].set_ylabel("Peak width σ (nm)")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / 'size_vs_lipid_fraction.png', dpi=300)
    plt.show()


# --- Usage ---
df = load_surfactant_results_from_json("surfactants")
plot_peak_vs_concentration(df)
plot_peak_vs_fraction(df)