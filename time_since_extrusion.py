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
        return float(peak.get("mean_nm") or 0)
    except Exception:
        return None


def extract_peak_width(entry):
    """Take width from peak with largest area."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None
    peak = max(peaks, key=lambda x: float(x.get("area_percent") or 0))
    try:
        return float(peak.get("size_peak_nm") or 0)
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
def time_series_plot(extractor, ylabel):
    plt.figure()
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
                day_values[dt].append(value)
            except Exception:
                continue

        if not day_values:
            continue

        days_sorted = sorted(day_values.keys())
        elapsed_days = [(day - global_date0).days for day in days_sorted]
        means = [np.mean(day_values[day]) for day in days_sorted]

        plt.plot(elapsed_days, means, marker="o", label=f"{t} °C")

    plt.xlabel("Time since extrusion (days)")
    plt.ylabel(ylabel)
    plt.legend(title="Temperature")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"{ylabel.replace(' ', '_')}_extrusion_time.png", dpi=300)
    plt.show()


# ----------------------------
# generate plots
# ----------------------------
time_series_plot(extractor=extract_peak_diameter, ylabel="Peak Diameter (nm)")
time_series_plot(extractor=extract_peak_width, ylabel="Peak Width σ (nm)")
