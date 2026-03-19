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
from datetime import datetime

# ----------------------------
# Load measurements from JSON
# ----------------------------
def load_measurements(folder="POPC"):
    """
    Load all JSON data from the folder.
    Extract temperature and extrusion from sample_name.
    Keep only the latest entry per sample_name.
    Choose peak with largest area.
    """
    folder_path = DATA_FOLDER / folder

    latest_entries = {}

    for file_path in folder_path.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            sample_name = entry.get("sample_name", "")
            sample_name = re.sub(r"\bnew_sample\b\s*", "", sample_name)
            sample_name = re.sub(r"\s+", " ", sample_name).strip()
            timestamp_str = entry.get("timestamp")

            if not timestamp_str:
                continue  # skip if no timestamp

            # parse timestamp (adjust format if needed)
            timestamp = datetime.strptime(timestamp_str, "%d %B %Y %H:%M:%S")

            # keep only latest entry per sample_name
            if (
                sample_name not in latest_entries
                or timestamp > latest_entries[sample_name]["_parsed_timestamp"]
            ):
                entry["_parsed_timestamp"] = timestamp
                latest_entries[sample_name] = entry

    # now process only deduplicated entries
    entries = []

    for entry in latest_entries.values():
        sample_name = entry.get("sample_name", "")

        # Extract extrusion
        extrusion_match = re.search(r"(\d+)\s+Extrusion", sample_name)
        entry["extrusion"] = int(extrusion_match.group(1)) if extrusion_match else 0

        # Extract temperature
        temp_match = re.search(r"(\d+)\s+degrees", sample_name)
        entry["temperature_C"] = float(temp_match.group(1)) if temp_match else 0

        # pick peak with largest area
        if entry.get("peaks"):
            peak = max(entry["peaks"], key=lambda x: float(x.get("area_percent") or 0))
            entry["peak_size_nm"] = float(peak.get("peak_position_nm") or 0)
            entry["peak_sigma_nm"] = float(peak.get("peak_width_nm") or 0)

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
# Compute stats
# ----------------------------
def compute_group_stats(all_data, control, independent, extractor):
    stats = {}

    for c in control:
        means, errors = [], []

        for i in independent:
            data = [
                d for d in all_data
                if d["temperature_C"] == i and d["extrusion"] == c
            ]
            values = extractor(data)

            if len(values) >= 2:
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1))
            else:
                means.append(np.nan)
                errors.append(np.nan)

        stats[c] = {
            "means": means,
            "errors": errors,
            "overall_mean": np.nanmean(means),
            "overall_std": np.nanstd(means),
        }

    return stats


# ----------------------------
# Plot grouped bars
# ----------------------------
def plot_grouped_bars(ax, stats, control, independent, ylabel):
    x = np.arange(len(independent))
    bar_width = 0.8 / len(control)

    for idx, c in enumerate(control):
        means = stats[c]["means"]
        errors = stats[c]["errors"]

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


# ----------------------------
# Plot overall trend
# ----------------------------
def plot_trend_at_temperature(ax, all_data, control, temperature, extractor, ylabel):
    means, errors = [], []

    for c in control:
        data = [
            d for d in all_data
            if d["temperature_C"] == temperature and d["extrusion"] == c
        ]

        values = extractor(data)

        if len(values) >= 2:
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1))
        else:
            means.append(np.nan)
            errors.append(np.nan)

    ax.errorbar(
        control,
        means,
        yerr=errors,
        fmt='o-',
        linewidth=2,
        markersize=6,
        capsize=5,
    )

    ax.set_xlabel("Number of Extrusions")
    ax.set_ylabel(f"{ylabel} (at {temperature}°C)")
    ax.grid(linestyle="--", alpha=0.3)


def grouped_bar_plot(control, independent, extractor, ylabel, filename, folder="POPC"):
    all_data = load_measurements(folder)

    stats = compute_group_stats(all_data, control, independent, extractor)

    # --- grouped bars ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_grouped_bars(ax, stats, control, independent, ylabel)

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{filename}_Temp_plot.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_trend_at_temperature(ax, all_data, control, 30, extractor, ylabel)
    
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{filename}_extrusion_30C.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    extrusions = [3, 5, 10, 15, 20, 31, 41, 51, 61]
    temperatures = [10, 20, 30, 40, 50, 60]

    grouped_bar_plot(
        extrusions,
        temperatures,
        extractor=extract_peak_diameters,
        ylabel="Peak Diameter (nm)",
        filename="Diameter",
        folder="POPC"
    )

    grouped_bar_plot(
        extrusions,
        temperatures,
        extractor=extract_sigmas,
        ylabel="Peak Width (nm)",
        filename="Sigma",
        folder="POPC"
    )