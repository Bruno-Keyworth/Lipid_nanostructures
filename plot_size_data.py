#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:52:29 2026

@author: brunokeyworth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

# ============================================================
# DATA LOADING
# ============================================================

def load_measurements_by_lipid(folder, lipids_of_interest=None):
    """
    Walk subfolders named like '30_degrees', read JSONs, and return a dict of DataFrames keyed by lipid.
    Only the dominant lipid (ratio 10) is extracted.
    """
    lipid_entries = {}
    
    folder = DATA_FOLDER / folder

    for subfolder in folder.glob("*_degrees"):
        temp_match = re.match(r"(\d+)_degrees", subfolder.name)
        if not temp_match:
            continue
        temperature = float(temp_match.group(1))

        for file in subfolder.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            for entry in data:
                if not isinstance(entry, dict):
                    continue

                base_name = entry.get("base_sample_name")
                if not base_name:
                    continue
                base_name = re.sub(r"\s+", " ", base_name).strip()
                entry["sample_name"] = base_name

                entry["extrusion"] = entry.get("extrusions", 0)
                entry["temperature_C"] = entry.get("temperature_C", temperature)

                lipid_ratio = entry.get("lipid_ratio", {})
                dominant_lipids = [L for L, v in lipid_ratio.items() if v == 10]
                if not dominant_lipids:
                    continue
                dominant_lipid = dominant_lipids[0]
                if lipids_of_interest and dominant_lipid not in lipids_of_interest:
                    continue
                entry["lipid"] = dominant_lipid

                averaged_peaks = entry.get("averaged_peaks", [])
                if averaged_peaks:
                    first_peak = averaged_peaks[0]
                    entry["peak_size_nm"] = first_peak.get("peak_position_nm", [0])[0]  # take mean value
                    entry["peak_sigma_nm"] = first_peak.get("peak_width_nm", [0])[0]
                    entry["peak_area_percent"] = first_peak.get("area_percent", [0])[0]
                else:
                    entry["peak_size_nm"] = 0
                    entry["peak_sigma_nm"] = 0
                    entry["peak_area_percent"] = 0

                entry["folder_temperature"] = temperature

                if dominant_lipid not in lipid_entries:
                    lipid_entries[dominant_lipid] = []
                lipid_entries[dominant_lipid].append(entry)

    # convert to DataFrames
    for lipid in lipid_entries:
        lipid_entries[lipid] = pd.DataFrame(lipid_entries[lipid])

    return lipid_entries

# ============================================================
# DATA EXTRACTORS
# ============================================================

def extract_peak_diameters(data):
    return [d["peak_size_nm"] for d in data if d.get("peak_size_nm", 0) > 0]

def extract_sigmas(data):
    return [d["peak_sigma_nm"] for d in data if d.get("peak_sigma_nm", 0) > 0]

def model(n, D_inf, D0, N):
    return D_inf + (D0 - D_inf) * np.exp(-n / N)

# ============================================================
# MULTIPLE LIPIDS PLOTTING
# ============================================================

def plot_trend_multi_lipid(ax, lipid_dfs, lipids, extractor, ylabel, FIT=True):
    for lipid in lipids:
        df = lipid_dfs.get(lipid)
        if df is None or df.empty:
            print(f"Warning: no data found for {lipid}")
            continue

        # auto-detect extrusion numbers
        extrusions_lipid = sorted(df["extrusion"].unique())
        means, errors = [], []

        for c in extrusions_lipid:
            subset = df[df["extrusion"] == c].to_dict("records")
            values = extractor(subset)
            if len(values) >= 2:
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1))
            else:
                means.append(np.nan)
                errors.append(np.nan)

        # fit
        if FIT:
            initial_guess = [100, 300, 15]
            try:
                params, cov = curve_fit(
                    model,
                    extrusions_lipid,
                    means,
                    p0=initial_guess,
                    sigma=errors,
                    absolute_sigma=True
                )
                n_fit = np.linspace(min(extrusions_lipid), max(extrusions_lipid), 200)
                ax.plot(n_fit, model(n_fit, *params), label=f"{lipid} fit")
            except Exception as e:
                print(f"Fit failed for {lipid}: {e}")

        ax.errorbar(
            extrusions_lipid,
            means,
            yerr=errors,
            fmt='o-',
            capsize=5,
            label=lipid
        )

    ax.set_xlabel("Number of Extrusions")
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", alpha=0.3)

def multi_lipid_plot(lipids, extractor, ylabel, filename, folder=DATA_FOLDER):
    lipid_dfs = load_measurements_by_lipid(folder, lipids_of_interest=lipids)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trend_multi_lipid(ax, lipid_dfs, lipids, extractor, ylabel)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{filename}_multi_lipid.png", dpi=300)
    plt.show()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    plt.close('all')

    lipids = ["POPC", "DMPC"]

    multi_lipid_plot(
        lipids,
        extractor=extract_peak_diameters,
        ylabel="Peak Diameter (nm)",
        filename="Diameter",
        folder="POPC"
    )

    multi_lipid_plot(
        lipids,
        extractor=extract_sigmas,
        ylabel="Peak Width (nm)",
        filename="Sigma",
        folder= "POPC"
    )