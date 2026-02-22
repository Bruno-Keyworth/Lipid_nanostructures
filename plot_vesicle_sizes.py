#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:03:47 2026

@author: brunokeyworth
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import get_file, PLOTS_FOLDER, DATA_FOLDER

extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

for e in extrusions:
    temps= []
    peaks = []
    errors = []

    for t in temperatures:
        file = get_file(t, e)

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            fallback_path = os.path.join(DATA_FOLDER, "unrecorded_data.txt")
            try:
                fallback = np.genfromtxt(
                    fallback_path,
                    delimiter=",",
                    names=True,
                    dtype=None,
                    encoding="utf-8"
                )

                mask = (
                    (fallback["Extrusion"] == e) &
                    (fallback["Temp"] == t)
                )

                selected = fallback[mask]

                # Build synthetic data structure matching expected JSON format
                data = []
                for row in selected:
                    if not np.isnan(row["Mean_nm"]):
                        data.append({
                            "meta": {
                                "CONTIN Peaks[1]": row["Mean_nm"]
                            }
                        })

            except Exception:
                continue
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
        ax1.errorbar(temps, peaks, yerr=errors, marker="o", ls='', label=f"{e} extrusions")


ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Peak Diameter (nm)")
ax1.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / 'Diameter_temperature_plot.png', dpi=300)


for t in temperatures:
    extr= []
    peaks = []
    errors = []

    for e in extrusions:
        file = get_file(t, e)

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            fallback_path = os.path.join(DATA_FOLDER, "unrecorded_data.txt")
            try:
                fallback = np.genfromtxt(
                    fallback_path,
                    delimiter=",",
                    names=True,
                    dtype=None,
                    encoding="utf-8"
                )

                mask = (
                    (fallback["Extrusion"] == e) &
                    (fallback["Temp"] == t)
                )

                selected = fallback[mask]

                # Build synthetic data structure matching expected JSON format
                data = []
                for row in selected:
                    if not np.isnan(row["Mean_nm"]):
                        data.append({
                            "meta": {
                                "CONTIN Peaks[1]": row["Mean_nm"]
                            }
                        })

            except Exception:
                continue
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
            extr.append(e)
            peaks.append(np.mean(values))
            errors.append(np.std(values))

    if extr:
        ax2.errorbar(extr, peaks, yerr=errors, marker="o", ls='', label=f"{t} °C")

ax2.set_xlabel("Number of Extrusions")
ax2.set_ylabel("Peak Diameter (nm)")
ax2.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / 'Diameter_extrusions_plot.png', dpi=300)
plt.show()



def plot_sizes(control, independent, control_label, xlabel):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for c in control:
        ind= []
        peaks = []
        errors = []
    
        for i in independent:
            if xlabel == "Temp, [°C]":
                file = get_file(i, c)
            else:
                file = get_file(c, i)  
                
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                fallback_path = DATA_FOLDER / "unrecorded_data.txt"
                try:
                    fallback = np.genfromtxt(
                        fallback_path,
                        delimiter=",",
                        names=True,
                        dtype=None,
                        encoding="utf-8"
                    )
    
                    if xlabel == "Temp, [°C]":
                        mask = (
                            (fallback["Extrusion"] == c) &
                            (fallback["Temp"] == i)
                        )
                    else:
                        mask = (
                            (fallback["Extrusion"] == i) &
                            (fallback["Temp"] == c))
    
                    selected = fallback[mask]
    
                    # Build synthetic data structure matching expected JSON format
                    data = []
                    for row in selected:
                        if not np.isnan(row["Mean_nm"]):
                            data.append({
                                "meta": {
                                    "CONTIN Peaks[1]": row["Mean_nm"]
                                }
                            })
    
                except Exception:
                    continue
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
                ind.append(i)
                peaks.append(np.mean(values))
                errors.append(np.std(values))
    
        if ind:
            ax.errorbar(ind, peaks, yerr=errors, marker="o", label = f"{c}{control_label}", 
                        linestyle='', markeredgecolor='black', capsize = 4,
                        markersize=6, elinewidth=1.2, markeredgewidth=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak Diameter (nm)")
    ax.legend()
    plt.tight_layout()
    filename = f"Diameter_{xlabel}_plot.png"
    plt.savefig(PLOTS_FOLDER / filename, dpi=300)
    plt.show()
    
plot_sizes(temperatures, extrusions, "°C", "Number of Extrusions")

plot_sizes(extrusions, temperatures, " Extrusions", "Temp, [°C]")

