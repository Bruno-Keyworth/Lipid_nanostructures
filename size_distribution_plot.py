#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:39:49 2026

@author: brunokeyworth
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

# ---- Load JSON file ----
file_path = DATA_FOLDER / 'surfactants' / '7_DMPC__3_DMPG_0.1_mg_ml_C12E6_100_microM.json'

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

ax.plot(sizes, intensities, marker='o', markersize=3, c='b', label='DLS Intensity Distribution')

# Peak position
ax.axvline(
    peak_pos,
    linestyle="--",
    label=f"Peak position = {peak_pos:.1f} nm",
    c='r'
)

# Peak width region
left = peak_pos - peak_width / 2
right = peak_pos + peak_width / 2

# Example schematic width region
width_left = 55
width_right = 140

# Height where the arrow will sit
arrow_y = 8

# Double-headed arrow
ax.annotate(
    '',
    xy=(width_left, arrow_y),
    xytext=(width_right, arrow_y),
    arrowprops=dict(arrowstyle='<->', lw=2)
)

# Label
ax.text(
    (width_left + width_right) / 2,
    arrow_y + 0.8,
    'Peak Width',
    ha='center',
    fontsize=20
)
ax.tick_params(labelsize=18)
ax.set_xscale("log")
ax.set_xlim(30, 300)

ax.set_xlabel("Hydrodynamic Diameter (nm)", fontsize=22)
ax.set_ylabel("Relative Intensity (%)", fontsize=22)

ax.legend(fontsize=20, framealpha=0)

plt.tight_layout()
plt.savefig(PLOTS_FOLDER / 'size_distribution.png', dpi=300)
plt.show()