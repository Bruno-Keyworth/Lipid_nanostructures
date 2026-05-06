#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:52:29 2026

@author: brunokeyworth
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER
import re

def load_measurements(parent_folder="extrusions"):
    """
    Recursively loads JSON files from parent_folder.
    Filters for pure POPC (10) or pure DMPC (10) using lipid_ratio.
    """
    folder_path = DATA_FOLDER / parent_folder
    processed_entries = []

    # Walk through all subfolders (30_degrees, 50_degrees, etc.)
    for file_path in folder_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # If the JSON is a list, take the first element; if a dict, use as is
                entries = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue

        for entry in entries:
            ratios = entry.get("lipid_ratio", {})

            # Determine Lipid Type based on ratio of 10
            lipid_type = None
            if ratios.get("POPC") == 10:
                lipid_type = "POPC"
            elif ratios.get("DMPC") == 10:
                lipid_type = "DMPC"
            elif ratios.get("DOPC") == 10:
                lipid_type = "DOPC"

            # Skip if it's not one of our target pure lipids
            if not lipid_type:
                continue

            # Extract Metadata
            temp = entry.get("temperature_C", 0)
            extrusions = entry.get("extrusions", 0)

            # Extract Peaks from repeat_peaks (flattening the list of lists)
            peaks = entry.get("repeat_peaks", [])
            for repeat in peaks:
                if not repeat:
                    continue

                # pick largest peak in this measurement (repeat)
                best_peak = max(repeat, key=lambda x: x.get("area_percent", 0))

                processed_entries.append({
                    "lipid": lipid_type,
                    "temperature_C": float(temp),
                    "extrusions": int(extrusions),
                    "peak_size_nm": float(best_peak.get("peak_position_nm", np.nan)),
                    "peak_sigma_nm": float(best_peak.get("peak_width_nm", np.nan))
                })

    return processed_entries

def model(n, D_inf, D0, N):
    return D_inf + (D0 - D_inf) * np.exp(-n / N)

def add_series(ax, data, lipid, target_temp, extrusions_list, key, color):
    """Filters data for a series and adds points + fit to the plot."""
    means, errors = [], []
    
    subset = [d for d in data if d["lipid"] == lipid and d["temperature_C"] == target_temp]

    for e in extrusions_list:
        
        error_percent = 0.05
        vals = [
            d[key] for d in subset
            if d["extrusions"] == e and np.isfinite(d[key]) and d[key] > 0
        ]
        
        if len(vals) > 1:
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            if std == 0:
                err = max(std, error_percent * mean)
            else:
                err = std
            
        elif len(vals) == 1:
            
            mean = vals[0]
            err = error_percent * mean
            
        else:
            mean = np.nan
            err = np.nan

        means.append(mean)
        errors.append(err)
        #print(f"{lipid}, extrusions = {e}, mean = {means[-1]}, err = {errors[-1]}")
        

    means, errors = np.array(means), np.array(errors)
    x_data = np.array(extrusions_list)
    mask = (~np.isnan(means)) & (~np.isnan(errors)) & (errors > 0)
    
    if np.any(mask):
        # Plot data points
        ax.errorbar(x_data[mask], means[mask], yerr=errors[mask], fmt='o', 
                    color=color, label=f"{lipid} ({target_temp}°C)", capsize=4)
        
        # Perform Fit
        try:
            popt, pcov = curve_fit(model, x_data[mask], means[mask], 
                               p0=[min(means[mask]), max(means[mask]), 15], 
                               sigma=errors[mask], absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            
            y_fit = model(x_data[mask], *popt)
            residuals = means[mask] - y_fit
            chi2 = np.sum((residuals / errors[mask])**2)
            dof = len(y_fit) - len(popt)
            chi2_red = chi2 / dof if dof > 0 else np.nan
        
            # Print results
            print(f"\n{lipid} at {target_temp}°C, {key}")
            print(f"D_inf = {popt[0]:.3f} ± {perr[0]:.3f}")
            print(f"D0    = {popt[1]:.3f} ± {perr[1]:.3f}")
            print(f"N     = {popt[2]:.3f} ± {perr[2]:.3f}")
            print(f"Reduced chi^2 = {chi2_red:.3f}")

            x_fit = np.linspace(min(extrusions_list), max(extrusions_list), 100)
            ax.plot(x_fit, model(x_fit, *popt), '--', color=color, alpha=0.7)
        except Exception as e:
            print(f"Fit failed for {lipid} {target_temp}C: {e}")

def generate_comparison(extrusions_list, key, ylabel, filename):
    all_data = (
    load_measurements("extrusions") +
    load_measurements("data_from_kate")
)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Target: POPC at 30 degrees and DMPC at 50 degrees
    add_series(ax, all_data, "POPC", 30.0, extrusions_list, key, "royalblue")
    add_series(ax, all_data, "DMPC", 50.0, extrusions_list, key, "firebrick")
    add_series(ax, all_data, "DOPC", 25.0, extrusions_list, key, "forestgreen")

    ax.set_xlabel("Number of Extrusions")
    ax.set_ylabel(ylabel)
    #ax.set_title(f"Comparison of POPC (30°C) and DMPC (50°C)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"Comparison_{filename}.png", dpi=300)
    plt.show()
    
def generate_double_figure(extrusions_list, key1, key2, filename):
    all_data = (
    load_measurements("extrusions") +
    load_measurements("data_from_kate")
)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 5))
    
    # Target: POPC at 30 degrees and DMPC at 50 degrees
    add_series(ax1, all_data, "POPC", 30.0, extrusions_list, key1, "royalblue")
    add_series(ax1, all_data, "DMPC", 50.0, extrusions_list, key1, "firebrick")
    add_series(ax1, all_data, "DOPC", 25.0, extrusions_list, key1, "forestgreen")
    
    add_series(ax2, all_data, "POPC", 30.0, extrusions_list, key2, "royalblue")
    add_series(ax2, all_data, "DMPC", 50.0, extrusions_list, key2, "firebrick")
    add_series(ax2, all_data, "DOPC", 25.0, extrusions_list, key2, "forestgreen")

    ax1.set_xlabel("Number of Extrusions")
    ax2.set_xlabel("Number of Extrusions")
    ax1.set_ylabel("Peak Diameter, [nm]")
    ax2.set_ylabel("Peak Width, [nm]")
    fig.suptitle("Comparison of POPC (30°C) and DMPC (50°C)")
    ax1.legend()
    ax2.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"Comparison_{filename}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    extrusions = [3, 5, 10, 11, 15, 20, 21,  31, 41, 51, 61]
    
    # 1. Diameter Plot
    #generate_comparison(extrusions, "peak_size_nm", "Peak Diameter (nm)", "Diameter")
    
    # 2. Width (Sigma) Plot
    #generate_comparison(extrusions, "peak_sigma_nm", "Peak Width (nm)", "Width")
    
    generate_double_figure(extrusions, "peak_size_nm", "peak_sigma_nm" ,  "both")