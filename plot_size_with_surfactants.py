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


# -----------------------------
# Load JSON data
# -----------------------------

def load_surfactant_results_from_json(subfolder):

    folder_path = DATA_FOLDER / subfolder
    rows = []

    for file_path in folder_path.glob("*.json"):
        with open(file_path) as f:
            data = json.load(f)

        for entry in data:
            entry["filename"] = file_path.name
            rows.append(entry)

    df = pd.DataFrame(rows)

    if "sample_name" not in df.columns:
        raise ValueError("JSON files must contain 'sample_name' field")

    df["charged_fraction"] = df["sample_name"].apply(compute_charged_fraction)
    df["surfactant"] = df["sample_name"].apply(extract_surfactant)
    df["conc_microM"] = df["sample_name"].apply(extract_conc_microM)

    # Convert timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Keep most recent measurement per sample and type
        df = (
            df.sort_values("timestamp")
              .groupby(["sample_name", "type"], as_index=False)
              .tail(1)
        )

    return df


# -----------------------------
# Extract dominant peak
# -----------------------------

def extract_dominant_peaks(df):

    df = df.copy()

    peaks_df = df.explode("peaks")

    peaks_df["area_percent"] = peaks_df["peaks"].apply(
        lambda p: p.get("area_percent", np.nan) if isinstance(p, dict) else np.nan
    )

    peaks_df["peak_nm"] = peaks_df["peaks"].apply(
        lambda p: p.get("mean_nm", np.nan) if isinstance(p, dict) else np.nan
    )

    peaks_df["sigma_nm"] = peaks_df["peaks"].apply(
        lambda p: p.get("size_peak_nm", np.nan) if isinstance(p, dict) else np.nan
    )

    idx = peaks_df.groupby(peaks_df.index)["area_percent"].idxmax()

    return peaks_df.loc[idx].copy()


# -----------------------------
# Plot: size vs surfactant concentration
# -----------------------------

def plot_peak_vs_concentration(df, charged_fraction_filter=0.3):

    df_size = df[df["type"] == "size"].copy()

    df_size = df_size[np.isclose(df_size["charged_fraction"], charged_fraction_filter)]

    df_size = extract_dominant_peaks(df_size)

    stats = (
        df_size.groupby(["surfactant", "conc_microM"])
        .agg(mean_peak=("peak_nm", "mean"),
             std_peak=("peak_nm", "std"),
             mean_sigma=("sigma_nm", "mean"),
             std_sigma=("sigma_nm", "std"))
        .reset_index()
    )

    if stats.empty:
        print("No data available for this charged fraction.")
        return

    groups = stats["surfactant"].unique()

    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, surf in enumerate(groups):

        sub = stats[stats["surfactant"] == surf].sort_values("conc_microM")

        ax[0].errorbar(
            sub["conc_microM"], sub["mean_peak"],
            yerr=sub["std_peak"], marker="o",
            color=colors[i], label=surf
        )

        ax[1].errorbar(
            sub["conc_microM"], sub["mean_sigma"],
            yerr=sub["std_sigma"], marker="o",
            color=colors[i], label=surf
        )

    for axes in ax:
        axes.set_xlabel("Surfactant concentration (µM)")
        axes.set_xscale("log")
        axes.set_title("7 DMPC : 3 DMPG")

        handles, labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend()

    ax[0].set_ylabel("Peak size (nm)")
    ax[1].set_ylabel("Peak width σ (nm)")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "size_vs_surfactant_conc.png", dpi=300)
    plt.show()


# -----------------------------
# Plot: size vs charged fraction
# -----------------------------

def plot_peak_vs_fraction(df, fixed_conc=100):

    df_size = df[df["type"] == "size"].copy()

    df_fixed = df_size[np.isclose(df_size["conc_microM"], fixed_conc)].copy()

    df_fixed = extract_dominant_peaks(df_fixed)

    stats = (
        df_fixed.groupby(["surfactant", "charged_fraction"])
        .agg(mean_peak=("peak_nm", "mean"),
             std_peak=("peak_nm", "std"),
             mean_sigma=("sigma_nm", "mean"),
             std_sigma=("sigma_nm", "std"))
        .reset_index()
    )

    if stats.empty:
        print("No data available for this concentration.")
        return

    groups = stats["surfactant"].unique()

    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, surf in enumerate(groups):

        sub = stats[stats["surfactant"] == surf].sort_values("charged_fraction")

        ax[0].errorbar(
            sub["charged_fraction"], sub["mean_peak"],
            yerr=sub["std_peak"], marker="o",
            color=colors[i], label=surf
        )

        ax[1].errorbar(
            sub["charged_fraction"], sub["mean_sigma"],
            yerr=sub["std_sigma"], marker="o",
            color=colors[i], label=surf
        )

    for axes in ax:
        axes.set_xlabel("Charged lipid fraction")
        axes.set_title(f"Surfactant concentration = {fixed_conc} µM")
        axes.set_yscale("log")

        handles, labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend()

    ax[0].set_ylabel("Peak size (nm)")
    ax[1].set_ylabel("Peak width σ (nm)")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "size_vs_lipid_fraction.png", dpi=300)
    plt.show()


# -----------------------------
# Run analysis
# -----------------------------

df = load_surfactant_results_from_json("surfactants")

plot_peak_vs_concentration(df)
plot_peak_vs_fraction(df)