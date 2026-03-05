#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:24:44 2026

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
from get_standard_deviation import fit_gaussian

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
        encoding="utf-8",
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
# Measurement extractors
# -------------------------------------------------

def extract_peak_diameters(data):

    values = []

    for d in data:

        meta = d["meta"]

        if "CONTIN Peaks[1]" in meta and meta["CONTIN Peaks[1]"] not in ("", None):
            try:
                values.append(float(meta["CONTIN Peaks[1]"]))
            except Exception:
                pass

    return values


def extract_sigmas(data):

    sigmas = []

    for m in data:

        try:
            sizes = np.asarray(m["size"][0], float)
            intensities = np.asarray(m["size"][1], float)

            stacked = np.column_stack((sizes, intensities))

            sigma = fit_gaussian(stacked, PLOT=False)

            if not np.isnan(sigma):
                sigmas.append(sigma)

        except Exception:
            continue

    return sigmas


# -------------------------------------------------
# Determine global time origin
# -------------------------------------------------

def find_global_start():

    dates = []

    for t in temperatures:

        try:
            with open(_get_file(t, extrusion), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

        if data:

            for d in data:

                meta = d["meta"]

                if "Measurement Date and Time" in meta:

                    try:
                        dt = datetime.strptime(
                            meta["Measurement Date and Time"],
                            time_format
                        )

                        dates.append(dt.date())

                    except Exception:
                        pass

        else:

            fb = fallback_select(t, extrusion)

            if fb is not None and "Date" in fb.dtype.names:

                for d in fb["Date"]:

                    try:
                        dt = datetime.strptime(d, "%Y-%m-%d").date()
                        dates.append(dt)

                    except Exception:
                        pass

    if not dates:
        raise RuntimeError("No valid measurement dates found.")

    return min(dates)


global_date0 = find_global_start()


# -------------------------------------------------
# General time-series plot
# -------------------------------------------------

def time_series_plot(extractor, fallback_column, ylabel):

    plt.figure()

    for t in temperatures:

        day_values = defaultdict(list)

        try:
            with open(_get_file(t, extrusion), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

        if data:

            extracted = extractor(data)

            for m, value in zip(data, extracted):

                try:

                    dt = datetime.strptime(
                        m["meta"]["Measurement Date and Time"],
                        time_format
                    ).date()

                    day_values[dt].append(value)

                except Exception:
                    continue

        # fallback
        if not day_values:

            fb = fallback_select(t, extrusion)

            if fb is not None and "Date" in fb.dtype.names:

                for d, value in zip(fb["Date"], fb[fallback_column]):

                    try:

                        dt = datetime.strptime(d, "%Y-%m-%d").date()

                        day_values[dt].append(float(value))

                    except Exception:
                        continue

        if not day_values:
            continue

        days_sorted = sorted(day_values.keys())

        elapsed_days = [
            (day - global_date0).days
            for day in days_sorted
        ]

        means = [
            np.mean(day_values[day])
            for day in days_sorted
        ]

        plt.plot(
            elapsed_days,
            means,
            marker="o",
            label=f"{t} °C"
        )

    plt.xlabel("Time since first measurement (days)")
    plt.ylabel(ylabel)
    plt.legend(title="Temperature")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Generate plots
# -------------------------------------------------

time_series_plot(
    extractor=extract_peak_diameters,
    fallback_column="Mean_nm",
    ylabel="Peak Diameter (nm)",
)

time_series_plot(
    extractor=extract_sigmas,
    fallback_column="Sigma_nm",
    ylabel="Peak Width σ (nm)",
)