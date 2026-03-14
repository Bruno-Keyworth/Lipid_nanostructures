#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:37:15 2026

@author: brunokeyworth
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_filepaths import DATA_FOLDER 

DATA_DIR = DATA_FOLDER / 'surfactants' / '50_degrees'

records = []

for fp in DATA_DIR.glob("*.json"):
    with open(fp) as f:
        data = json.load(f)

    lipid = data["lipid_ratio"]
    surf = data["surfactant_conc_microM"]

    DMPC = lipid.get("DMPC", 0)
    DMPG = lipid.get("DMPG", 0)

    if DMPC == 0 and DMPG == 0:
        continue

    total = DMPC + DMPG
    frac_DMPG = DMPG / total if total > 0 else np.nan

    peaks = data.get("averaged_peaks", [])

    p1 = peaks[0] if len(peaks) > 0 else {}
    p2 = peaks[1] if len(peaks) > 1 else {}
    p3 = peaks[2] if len(peaks) > 2 else {}

    record = {
        "fraction_DMPG": frac_DMPG,

        "zeta": data["average_zeta"][0],

        "p1_pos": p1.get("peak_position_nm", [np.nan])[0] if p1 else np.nan,
        "p1_width": p1.get("peak_width_nm", [np.nan])[0] if p1 else np.nan,
        "p1_area": p1.get("area_percent", [np.nan])[0] if p1 else np.nan,

        "p2_pos": p2.get("peak_position_nm", [np.nan])[0] if p2 else np.nan,
        "p2_width": p2.get("peak_width_nm", [np.nan])[0] if p2 else np.nan,
        "p2_area": p2.get("area_percent", [np.nan])[0] if p2 else np.nan,
        
        "p3_pos": p3.get("peak_position_nm", [np.nan])[0] if p3 else np.nan,
        "p3_width": p3.get("peak_width_nm", [np.nan])[0] if p3 else np.nan,
        "p3_area": p3.get("area_percent", [np.nan])[0] if p3 else np.nan,


        "C12E6": surf.get("C12E6", 0),
        "DDAC": surf.get("DDAC", 0),
        "TX100": surf.get("TX100", 0),
    }

    records.append(record)

df = pd.DataFrame(records)

# --- classify surfactant condition ---
def surfactant_condition(row):
    if row["C12E6"] == 0 and row["DDAC"] == 0 and row["TX100"] == 0:
        return "Control"

    if row["C12E6"] == 100:
        return "C12E6 100 µM"
    if row["DDAC"] == 100:
        return "DDAC 100 µM"
    if row["TX100"] == 100:
        return "TX100 100 µM"

    return None

df["condition"] = df.apply(surfactant_condition, axis=1)
df = df.dropna(subset=["condition"])

conditions = ["Control", "C12E6 100 µM", "DDAC 100 µM", "TX100 100 µM"]

plots = {
    "zeta": "Zeta Potential (mV)",
    "p1_pos": "Peak 1 Position (nm)",
    "p1_width": "Peak 1 Width (nm)",
    "p1_area": "Peak 1 Area (%)",
    "p2_pos": "Peak 2 Position (nm)",
    "p2_width": "Peak 2 Width (nm)",
    "p2_area": "Peak 2 Area (%)",
    "p3_pos": "Peak 3 Position (nm)",
    "p3_width": "Peak 3 Width (nm)",
    "p3_area": "Peak 3 Area (%)",
}

for key, ylabel in plots.items():

    plt.figure()

    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")

        if len(sub) == 0:
            continue

        plt.plot(
            sub["fraction_DMPG"],
            sub[key],
            marker="o",
            label=cond,
        )

    plt.xlabel("Fraction DMPG / (DMPC + DMPG)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
