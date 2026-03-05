#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:52:29 2026

@author: brunokeyworth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from get_filepaths import _get_file, DATA_FOLDER

extrusion = 31
temperatures = [10, 20, 30, 40, 50, 60]
time_format = "%d %B %Y %H:%M:%S"

# -------------------------------------------------
# Load fallback data
# -------------------------------------------------
fallback_path = DATA_FOLDER / "unrecorded_data.txt"
try:
    fallback_data = np.genfromtxt(
        fallback_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )
except Exception:
    fallback_data = None


def fallback_select(temp, extrusion):
    if fallback_data is None:
        return None

    mask = (
        (fallback_data["Temp"] == temp) &
        (fallback_data["Extrusion"] == extrusion)
    )

    selected = fallback_data[mask]
    return selected if len(selected) > 0 else None


# -------------------------------------------------
# First pass: find global earliest measurement date
# -------------------------------------------------
all_dates = []

for t in temperatures:
    try:
        with open(_get_file(t, extrusion), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None

    if data is not None:
        for d in data:
            meta = d["meta"]
            if "Measurement Date and Time" in meta:
                try:
                    dt = datetime.strptime(
                        meta["Measurement Date and Time"],
                        time_format
                    )
                    all_dates.append(dt.date())
                except (ValueError, TypeError):
                    pass

    else:
        fb = fallback_select(t, extrusion)
        if fb is not None and "Date" in fb.dtype.names:
            for d in fb["Date"]:
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d").date()
                    all_dates.append(dt)
                except Exception:
                    pass

if not all_dates:
    raise RuntimeError("No valid measurement dates found.")

global_date0 = min(all_dates)

# -------------------------------------------------
# Second pass: plotting
# -------------------------------------------------
plt.figure()

for t in temperatures:

    try:
        with open(_get_file(t, extrusion), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None

    day_values = defaultdict(list)

    if data is not None:

        for d in data:
            meta = d["meta"]

            if (
                "CONTIN Peaks[1]" not in meta or
                "Measurement Date and Time" not in meta
            ):
                continue

            try:
                peak = float(meta["CONTIN Peaks[1]"])
                dt = datetime.strptime(
                    meta["Measurement Date and Time"],
                    time_format
                ).date()
            except (ValueError, TypeError):
                continue

            day_values[dt].append(peak)

    # ---------------- fallback ----------------
    if not day_values:
        fb = fallback_select(t, extrusion)

        if fb is not None and "Date" in fb.dtype.names:
            for d, peak in zip(fb["Date"], fb["Mean_nm"]):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d").date()
                    day_values[dt].append(float(peak))
                except Exception:
                    continue

    if not day_values:
        continue

    days_sorted = sorted(day_values.keys())

    elapsed_days = [
        (day - global_date0).days
        for day in days_sorted
    ]

    peaks_avg = [
        np.mean(day_values[day])
        for day in days_sorted
    ]

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