#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:03:47 2026

@author: brunokeyworth
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import get_file, PLOTS_FOLDER

extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]

plt.figure()

for e in extrusions:
    temps= []
    peaks = []
    errors = []

    for t in temperatures:
        file = get_file(t, e)

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            continue

        # Multiple measurements per file → average them
        values = [
            float(d["meta"]["CONTIN Peaks[1]"])
            for d in data
            if "CONTIN Peaks[1]" in d["meta"]
               and d["meta"]["CONTIN Peaks[1]"] not in ("", None)
        ]

        if values:               
            temps.append(t)
            peaks.append(np.mean(values))
            errors.append(np.std(values))

    if temps:
        plt.errorbar(temps, peaks, yerr=errors, marker="o", ls='', label=f"{e} extrusions")

plt.xlabel("Temperature (°C)")
plt.ylabel("Peak Diameter (nm)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / 'Diameter_temperature_plot.png', dpi=300)
plt.show()
