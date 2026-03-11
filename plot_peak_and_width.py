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
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER
import re

# ----------------------------
# Load measurements from JSON
# ----------------------------
def load_measurements(folder="POPC"):
    """
    Load all JSON data from the folder.
    Extract temperature and extrusion from sample_name.
    Choose peak with largest area.
    """
    folder_path = DATA_FOLDER / folder
    entries = []

    for file_path in folder_path.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            sample_name = entry.get("sample_name", "")

            # Extract extrusion from sample_name
            extrusion_match = re.search(r"(\d+)\s+Extrusion", sample_name)
            entry["extrusion"] = int(extrusion_match.group(1)) if extrusion_match else 0

            # Extract temperature from sample_name
            temp_match = re.search(r"(\d+)\s+degrees", sample_name)
            entry["temperature_C"] = float(temp_match.group(1)) if temp_match else 0

            # pick peak with largest area
            if entry.get("peaks"):
                peak = max(entry["peaks"], key=lambda x: float(x.get("area_percent") or 0))
                entry["peak_size_nm"] = float(peak.get("mean_nm") or 0)
                entry["peak_sigma_nm"] = float(peak.get("size_peak_nm") or 0)

            entries.append(entry)

    return entries

# ----------------------------
# Extractors
# ----------------------------
def extract_peak_diameters(data):
    return [d["peak_size_nm"] for d in data if "peak_size_nm" in d and d["peak_size_nm"] > 0]

def extract_sigmas(data):
    return [d["peak_sigma_nm"] for d in data if "peak_sigma_nm" in d and d["peak_sigma_nm"] > 0]

# ----------------------------
# Grouped bar plot
# ----------------------------
def grouped_bar_plot(control, independent, extractor, ylabel, filename, folder="POPC"):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(independent))
    bar_width = 0.8 / len(control)

    # Load all data once
    all_data = load_measurements(folder)

    for idx, c in enumerate(control):
        means, errors = [], []

        for i in independent:
            # filter by temperature and extrusion
            data = [d for d in all_data if d["temperature_C"] == i and d["extrusion"] == c]
            values = extractor(data)
            if len(values) >= 2:
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1))
            else:
                means.append(np.nan)
                errors.append(np.nan)

        offset = (idx - (len(control) - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=errors,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label=f"{c} Extrusions",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(independent)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / filename, dpi=300)
    plt.show()

# ----------------------------
# Usage
# ----------------------------
if __name__ == "__main__":
    extrusions = [3, 5, 10, 15, 20, 31, 41]
    temperatures = [10, 20, 30, 40, 50, 60]

    grouped_bar_plot(
        extrusions,
        temperatures,
        extractor=extract_peak_diameters,
        ylabel="Peak Diameter (nm)",
        filename="Diameter_Temp_plot.png",
        folder="POPC"
    )

    grouped_bar_plot(
        extrusions,
        temperatures,
        extractor=extract_sigmas,
        ylabel="Mean Standard Deviation (nm)",
        filename="Sigma_Temp_plot.png",
        folder="POPC"
    )