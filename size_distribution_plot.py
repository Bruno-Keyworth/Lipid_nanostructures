#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:39:49 2026

@author: brunokeyworth
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from get_filepaths import DATA_FOLDER

# ---- Load JSON file ----
file_path = DATA_FOLDER / 'extrusions' / '31_Extrusion_POPC_0.2_mg_ml_30_degrees_new_sample.json'

with open(file_path, "r") as f:
    data = json.load(f)

# ---- Representative measurement ----
measurement = data[0]

sizes = np.array(measurement["sizes_nm"])
intensities = np.array(measurement["intensities_percent"])

# Instrument-reported peak
peak = measurement["peaks"][0]

peak_pos = peak["peak_position_nm"]
peak_width = peak["peak_width_nm"]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(sizes, intensities, marker='o', markersize=3)

# Peak position
ax.axvline(
    peak_pos,
    linestyle="--",
    label=f"Peak position = {peak_pos:.1f} nm"
)

# Peak width region
left = peak_pos - peak_width / 2
right = peak_pos + peak_width / 2

ax.axvspan(
    left,
    right,
    alpha=0.2,
    label=f"Peak width = {peak_width:.1f} nm"
)

ax.set_xscale("log")
ax.set_xlim(30, 300)

ax.set_xlabel("Hydrodynamic diameter (nm)")
ax.set_ylabel("Relative intensity (%)")

ax.legend()

plt.tight_layout()
plt.show()