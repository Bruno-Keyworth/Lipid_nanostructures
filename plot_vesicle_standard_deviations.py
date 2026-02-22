# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:28:46 2026

@author: David Mawson

plot sigma against extrusion number
 for all temps
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import get_file, PLOTS_FOLDER, DATA_FOLDER
from get_standard_deviation import fit_gaussian

extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]
#temperatures = [10, 30, 60]
#temperatures = [20, 40, 50]

fig, ax = plt.subplots(figsize=(10, 6))

fallback_path = os.path.join(DATA_FOLDER, "unrecorded_data.txt")


# Load fallback file once
try:
    fallback_data = np.genfromtxt(
        fallback_path,
        delimiter=",",
        names=True,
        dtype=None,      
        encoding="utf-8")
    
except Exception:
    print("---")
    fallback_data = None

for t in temperatures:

    x_vals = []
    mean_stds = []
    std_errors = []

    for e in extrusions:

        std_values = []

        try:
            file = get_file(t, e)

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for measurement in data:

                sizes = np.array(measurement["size"][0], dtype=float)
                intensities = np.array(measurement["size"][1], dtype=float)

                stacked = np.column_stack((sizes, intensities))

                std_val = fit_gaussian(stacked, PLOT=False)

                if not np.isnan(std_val):
                    std_values.append(std_val)

        except Exception:
            pass

        if (len(std_values) == 0) and (fallback_data is not None):

            mask = (
                (fallback_data["Extrusion"] == e) &
                (fallback_data["Temp"] == t)
            )

            selected = fallback_data[mask]

            if len(selected) > 0:
                
                std_values = selected["Sigma_nm"]

        #Compute mean + uncertainty
        if len(std_values) >= 2:

            x_vals.append(e)
            mean_stds.append(np.mean(std_values))
            std_errors.append(np.std(std_values, ddof=1))
    

    if len(x_vals) > 0:

        plt.errorbar(
            x_vals,
            mean_stds,
            yerr=std_errors,
            marker="o",
            label=f"{t} °C",
            linestyle='', markeredgecolor='black', capsize = 4,
            markersize=6, elinewidth=1.2, markeredgewidth=0.5)

ax.set_xlabel("Number of Extrusions")
ax.set_ylabel("Mean Standard Deviation")
ax.grid(linestyle="--", alpha=0.3)
ax.legend(loc="best")
plt.savefig(PLOTS_FOLDER / 'POPC_sigma_vs_extrusions_plot.png', dpi=300)
plt.show()


def plot_sigma(control, independent, control_label, xlabel):
    fig, axis = plt.subplots(figsize=(10, 6))
    
    for c in control:

        x_vals = []
        mean_stds = []
        std_errors = []

        for i in independent:

            std_values = []

            try:
                if xlabel == "Temp, [°C]":
                    file = get_file(i, c)
                else:
                    file = get_file(c, i) 

                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for measurement in data:

                    sizes = np.array(measurement["size"][0], dtype=float)
                    intensities = np.array(measurement["size"][1], dtype=float)

                    stacked = np.column_stack((sizes, intensities))

                    std_val = fit_gaussian(stacked, PLOT=False)

                    if not np.isnan(std_val):
                        std_values.append(std_val)

            except Exception:
                pass

            if (len(std_values) == 0) and (fallback_data is not None):

                if xlabel == "Temp, [°C]":
                    mask = (
                        (fallback_data["Extrusion"] == c) &
                        (fallback_data["Temp"] == i)
                    )
                else:
                    mask = (
                        (fallback_data["Extrusion"] == i) &
                        (fallback_data["Temp"] == c))

                selected = fallback_data[mask]

                if len(selected) > 0:
                    
                    std_values = selected["Sigma_nm"]

            #Compute mean + uncertainty
            if len(std_values) >= 2:

                x_vals.append(i)
                mean_stds.append(np.mean(std_values))
                std_errors.append(np.std(std_values, ddof=1))
        

        if len(x_vals) > 0:

            axis.errorbar(
                x_vals,
                mean_stds,
                yerr=std_errors,
                marker="o",
                label=f"{c} {control_label}",
                linestyle='', markeredgecolor='black', capsize = 4,
                markersize=6, elinewidth=1.2, markeredgewidth=0.5)

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean Standard Deviation")
    axis.grid(linestyle="--", alpha=0.3)
    axis.legend(loc="best")
    filename = f"standard_deviation_{xlabel}_plot.png"
    plt.savefig(PLOTS_FOLDER / filename, dpi=300)
    plt.show()
    
plot_sigma(temperatures, extrusions, "°C", "Number of Extrusions")

plot_sigma(extrusions, temperatures, " Extrusions", "Temp, [°C]")

