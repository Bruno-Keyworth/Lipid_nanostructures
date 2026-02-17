#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:24:44 2026

@author: brunokeyworth
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- fix imports ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from get_filepaths import get_file

extrusion = 31
temperatures = [10, 20, 30, 40, 50, 60]
time_format = "%d %B %Y %H:%M:%S"

# -------------------------------------------------
# First pass: find global earliest measurement date
# -------------------------------------------------
all_dates = []

for t in temperatures:
    file = get_file(t, extrusion)
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        continue

    for d in data:
        meta = d["meta"]
        if "Measurement Date and Time" in meta:
            try:
                dt = datetime.strptime(meta["Measurement Date and Time"], time_format)
                all_dates.append(dt.date())  # only the date
            except (ValueError, TypeError):
                pass

if not all_dates:
    raise RuntimeError("No valid measurement dates found.")

global_date0 = min(all_dates)

# -----------------------
# Second pass: plotting
# -----------------------
plt.figure()

for t in temperatures:
    file = get_file(t, extrusion)

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        continue

    # Collect values per date
    day_values = defaultdict(list)

    for d in data:
        meta = d["meta"]
        if "CONTIN Peaks[1]" not in meta or "Measurement Date and Time" not in meta:
            continue
        try:
            peak = float(meta["CONTIN Peaks[1]"])
            dt = datetime.strptime(meta["Measurement Date and Time"], time_format).date()
        except (ValueError, TypeError):
            continue

        day_values[dt].append(peak)

    if not day_values:
        continue

    # Average per day and sort
    days_sorted = sorted(day_values.keys())
    elapsed_days = [(day - global_date0).days for day in days_sorted]
    peaks_avg = [np.mean(day_values[day]) for day in days_sorted]

    plt.plot(
        elapsed_days,
        peaks_avg,
        marker="o",
        label=f"{t} °C"
    )

plt.xlabel("Time since first measurement (days)")
plt.ylabel("CONTIN Peaks[1] (daily average)")
plt.legend(title="Temperature")
plt.tight_layout()
plt.show()
