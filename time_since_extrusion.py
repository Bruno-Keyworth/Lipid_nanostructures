#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:24:44 2026

@author: brunokeyworth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER
import re

POPC_FOLDER = DATA_FOLDER / "POPC"
extrusion = 31
temperatures = [10, 20, 30, 40, 50, 60]
time_format = "%d %B %Y %H:%M:%S"


# ----------------------------
# helpers
# ----------------------------
def extract_extrusion(sample_name):
    try:
        return int(sample_name.split()[0])
    except Exception:
        return None


def load_entries():
    entries = []
    for file in POPC_FOLDER.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            sample_name = d.get("sample_name")
            if extract_extrusion(sample_name) != extrusion:
                continue
            entries.append(d)
    return entries


def extract_peak_diameter(entry):
    """Take diameter from peak with largest area."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None
    peak = max(peaks, key=lambda x: float(x.get("area_percent") or 0))
    try:
        return float(peak.get("peak_position_nm") or 0)
    except Exception:
        return None


def extract_peak_width(entry):
    """Take width from peak with largest area."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None
    peak = max(peaks, key=lambda x: float(x.get("area_percent") or 0))
    try:
        return float(peak.get("peak_width_nm") or 0)
    except Exception:
        return None


# ----------------------------
# load data
# ----------------------------
all_entries = load_entries()


def find_global_start():
    dates = []
    for entry in all_entries:
        ts = entry.get("timestamp")
        if ts:
            try:
                dt = datetime.strptime(ts, time_format).date()
                dates.append(dt)
            except Exception:
                continue
    if not dates:
        raise RuntimeError("No valid timestamps found.")
    return min(dates)


global_date0 = find_global_start()


# ----------------------------
# plotting
# ----------------------------
def time_series_bar_plot(extractor, ylabel):
    """
    Generate a grouped bar chart of mean values over temperatures for each day
    since extrusion, similar to grouped_bar_plot style.
    """
    # Collect data: for each temperature, get mean per day
    day_values_per_temp = {}
    for t in temperatures:
        day_values = defaultdict(list)
        for entry in all_entries:
            if entry.get("temperature_C") != t:
                continue
            value = extractor(entry)
            if value is None:
                continue
            ts = entry.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.strptime(ts, time_format).date()
                elapsed_days = (dt - global_date0).days
                day_values[elapsed_days].append(value)
            except Exception:
                continue
        if day_values:
            day_values_per_temp[t] = day_values

    # Determine all days present across temperatures
    all_days = sorted(set(day for day_values in day_values_per_temp.values() for day in day_values))

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_days))
    bar_width = 0.8 / len(temperatures)

    for idx, t in enumerate(temperatures):
        means, errors = [], []
        day_values = day_values_per_temp.get(t, {})
        
        for day in all_days:
            values = day_values.get(day, [])
            
            if values:
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1))
            else:
                means.append(np.nan)
                errors.append(np.nan)
        offset = (idx - (len(temperatures) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=errors,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label=f"{t} °C",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_days)
    ax.set_xlabel("Time since extrusion (days)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Temperature")
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{ylabel.replace(' ', '_')}_extrusion_bar.png", dpi=300)
    plt.show()


# Usage
time_series_bar_plot(extractor=extract_peak_diameter, ylabel="Peak Diameter (nm)")
time_series_bar_plot(extractor=extract_peak_width, ylabel="Peak Width σ (nm)")