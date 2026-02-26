#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:52:29 2026

@author: brunokeyworth
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from get_filepaths import _get_file, PLOTS_FOLDER, DATA_FOLDER
from get_standard_deviation import fit_gaussian

extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]

# -------------------------------------------------
# Load fallback data once
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


def load_measurements(temp, extrusion):
    """Load JSON measurements or return None."""
    try:
        with open(_get_file(temp, extrusion), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fallback_select(temp, extrusion):
    if fallback_data is None:
        return None

    mask = (
        (fallback_data["Temp"] == temp) &
        (fallback_data["Extrusion"] == extrusion)
    )
    selected = fallback_data[mask]
    return selected if len(selected) > 0 else None

def extract_peak_diameters(data):
    values = [
        float(d["meta"]["CONTIN Peaks[1]"])
        for d in data
        if "CONTIN Peaks[1]" in d["meta"]
           and d["meta"]["CONTIN Peaks[1]"] not in ("", None)
    ]
    return values


def extract_sigmas(data):
    sigmas = []
    for m in data:
        sizes = np.asarray(m["size"][0], float)
        intensities = np.asarray(m["size"][1], float)
        stacked = np.column_stack((sizes, intensities))
        sigma = fit_gaussian(stacked, PLOT=False)
        if not np.isnan(sigma):
            sigmas.append(sigma)
    return sigmas

def grouped_bar_plot(
    control,
    independent,
    extractor,
    fallback_column,
    ylabel,
    filename,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(independent))
    bar_width = 0.8 / len(control)

    for idx, c in enumerate(control):

        means, errors = [], []

        for i in independent:
            data = load_measurements(i, c)

            if data is not None:
                values = extractor(data)
            else:
                values = []

            # fallback
            if len(values) == 0:
                fb = fallback_select(i, c)
                if fb is not None:
                    values = fb[fallback_column]

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
    ax.set_xlabel("Temp, [°C]")
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(means)-20)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / filename, dpi=300)
    plt.show()
    
grouped_bar_plot(
    extrusions,
    temperatures,
    extractor=extract_peak_diameters,
    fallback_column="Mean_nm",
    ylabel="Peak Diameter (nm)",
    filename="Diameter_Temp_plot.png",
)

grouped_bar_plot(
    extrusions,
    temperatures,
    extractor=extract_sigmas,
    fallback_column="Sigma_nm",
    ylabel="Mean Standard Deviation (nm)",
    filename="Sigma_Temp_plot.png",
)