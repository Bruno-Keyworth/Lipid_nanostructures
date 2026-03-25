#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:52:29 2026

@author: brunokeyworth
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

def load_measurements_by_lipid(folder, lipids_temperatures={'POPC': 30, 'DMPC': 50}):
    """
    Walk subfolders like '30_degrees', load JSONs, extract:
        - extrusion number
        - temperature
        - dominant lipid (ratio 10)
        - largest-area averaged peak
    Return dict: lipid → DataFrame
    """

    folder = DATA_FOLDER / folder
    lipid_entries = {}

    for subfolder in folder.glob("*_degrees"):
        temp_match = re.match(r"(\d+)_degrees", subfolder.name)
        if not temp_match:
            continue

        folder_temp = float(temp_match.group(1))

        for json_file in subfolder.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Standardise to list
            if isinstance(data, dict):
                data = [data]

            for entry in data:
                if not isinstance(entry, dict):
                    continue

                # --- extraction of basic fields ---
                base_name = entry.get("base_sample_name", "")
                base_name = re.sub(r"\s+", " ", base_name).strip()
                entry["sample_name"] = base_name

                entry["extrusion"] = int(entry.get("extrusions", 0))
                entry["temperature_C"] = float(entry.get("temperature_C", folder_temp))

                # --- identify dominant lipid ---
                lipid_ratio = entry.get("lipid_ratio", {})
                dominant_candidates = [L for L, v in lipid_ratio.items() if v == 10]
                if not dominant_candidates:
                    continue

                dominant_lipid = dominant_candidates[0]

                # temperature filter per lipid
                if dominant_lipid in lipids_temperatures:
                    required_temp = lipids_temperatures[dominant_lipid]
                    if int(entry["temperature_C"]) != required_temp:
                        continue
                else:
                    continue

                entry["lipid"] = dominant_lipid

                # --- pick largest-area averaged peak (old script behaviour) ---
                averaged_peaks = entry.get("averaged_peaks", [])
                if averaged_peaks:
                    peak = max(averaged_peaks, key=lambda p: p["area_percent"][0])
                    entry["peak_size_nm"] = float(peak["peak_position_nm"][0])
                    entry["peak_sigma_nm"] = float(peak["peak_width_nm"][0])
                    entry["peak_area_percent"] = float(peak["area_percent"][0])
                else:
                    entry["peak_size_nm"] = np.nan
                    entry["peak_sigma_nm"] = np.nan
                    entry["peak_area_percent"] = np.nan

                # --- accumulate ---
                if dominant_lipid not in lipid_entries:
                    lipid_entries[dominant_lipid] = []

                lipid_entries[dominant_lipid].append(entry)

    # convert lists → DataFrames
    for lipid in lipid_entries:
        lipid_entries[lipid] = pd.DataFrame(lipid_entries[lipid])

    return lipid_entries

# ============================================================
# DATA EXTRACTORS
# ============================================================

def extract_peak_diameters(records):
    return [d["peak_size_nm"] for d in records if not pd.isna(d["peak_size_nm"])]

def extract_sigmas(records):
    return [d["peak_sigma_nm"] for d in records if not pd.isna(d["peak_sigma_nm"])]

def model(n, D_inf, D0, N):
    return D_inf + (D0 - D_inf) * np.exp(-n / N)

# ============================================================
# MULTI-LIPID TREND PLOTTING
# ============================================================

def plot_trend_multi_lipid(ax, lipid_dfs, lipids, extractor, ylabel, FIT=True):
    for lipid in lipids:
        df = lipid_dfs.get(lipid)
        if df is None or df.empty:
            print(f"Warning: no data for {lipid}")
            continue

        extrusions = sorted(df["extrusion"].unique())
        means, errors = [], []

        for n in extrusions:
            subset = df[df["extrusion"] == n].to_dict("records")
            values = extractor(subset)

            if len(values) >= 1:
                means.append(np.mean(values))
                if len(values) > 1:
                    errors.append(np.std(values, ddof=1))
                else:
                    errors.append(0)
            else:
                means.append(np.nan)
                errors.append(np.nan)

        # fit
        if FIT:
            try:
                params, cov = curve_fit(
                    model,
                    extrusions,
                    means,
                    p0=[100, 300, 15],
                    sigma=[e if e > 0 else 1 for e in errors],
                    absolute_sigma=True,
                    maxfev=10000
                )
                n_fit = np.linspace(min(extrusions), max(extrusions), 300)
                ax.plot(n_fit, model(n_fit, *params), label=f"{lipid} fit")
            except Exception as e:
                print(f"Fit failed for {lipid}: {e}")

        ax.errorbar(
            extrusions,
            means,
            yerr=errors,
            fmt='o-',
            capsize=5,
            label=lipid
        )

    ax.set_xlabel("Number of Extrusions")
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", alpha=0.3)

# ============================================================
# WRAPPER
# ============================================================

def multi_lipid_plot(extractor, ylabel, filename, folder):
    lipid_dfs = load_measurements_by_lipid(folder)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trend_multi_lipid(ax, lipid_dfs, ['POPC', 'DMPC'], extractor, ylabel)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{filename}_multi_lipid.png", dpi=300)
    plt.show()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    plt.close('all')

    multi_lipid_plot(
        extractor=extract_peak_diameters,
        ylabel="Peak Diameter (nm)",
        filename="Diameter",
        folder="POPC"
    )

    multi_lipid_plot(
        extractor=extract_sigmas,
        ylabel="Peak Width (nm)",
        filename="Sigma",
        folder="POPC"
    )