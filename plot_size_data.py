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

def load_measurements(parent_folder="POPC"):
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
            
            # Skip if it's not one of our target pure lipids
            if not lipid_type:
                continue

            # Extract Metadata
            temp = entry.get("temperature_C", 0)
            extrusions = entry.get("extrusions", 0)

            # Extract Peaks from repeat_peaks (flattening the list of lists)
            all_repeats = [p for sublist in entry.get("repeat_peaks", []) for p in sublist]
            
            if all_repeats:
                # Pick the peak with the largest area percent across all repeats
                best_peak = max(all_repeats, key=lambda x: x.get("area_percent", 0))
                
                processed_entries.append({
                    "lipid": lipid_type,
                    "temperature_C": float(temp),
                    "extrusions": int(extrusions),
                    "peak_size_nm": float(best_peak.get("peak_position_nm", 0)),
                    "peak_sigma_nm": float(best_peak.get("peak_width_nm", 0))
                })

    return processed_entries

def model(n, D_inf, D0, N):
    return D_inf + (D0 - D_inf) * np.exp(-n / N)

def add_series(ax, data, lipid, target_temp, extrusions_list, key, color):
    """Filters data for a series and adds points + fit to the plot."""
    means, errors = [], []
    
    subset = [d for d in data if d["lipid"] == lipid and d["temperature_C"] == target_temp]

    for e in extrusions_list:
        # Find all measurements for this specific extrusion count
        vals = [d[key] for d in subset if d["extrusions"] == e and d[key] > 0]
        
        if len(vals) >= 1:
            means.append(np.mean(vals))
            # If only 1 measurement, use 5% of mean as a placeholder error for fitting
            errors.append(np.std(vals, ddof=1) if len(vals) > 1 else np.mean(vals) * 0.05)
        else:
            means.append(np.nan)
            errors.append(np.nan)

    means, errors = np.array(means), np.array(errors)
    x_data = np.array(extrusions_list)
    mask = ~np.isnan(means)
    
    if np.any(mask):
        # Plot data points
        ax.errorbar(x_data[mask], means[mask], yerr=errors[mask], fmt='o', 
                    color=color, label=f"{lipid} ({target_temp}°C)", capsize=4)
        
        # Perform Fit
        try:
            popt, _ = curve_fit(model, x_data[mask], means[mask], 
                               p0=[min(means[mask]), max(means[mask]), 15], 
                               sigma=errors[mask], absolute_sigma=True)
            
            x_fit = np.linspace(min(extrusions_list), max(extrusions_list), 100)
            ax.plot(x_fit, model(x_fit, *popt), '--', color=color, alpha=0.7)
        except Exception as e:
            print(f"Fit failed for {lipid} {target_temp}C: {e}")

def generate_comparison(extrusions_list, key, ylabel, filename):
    all_data = load_measurements("POPC")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Target: POPC at 30 degrees and DMPC at 50 degrees
    add_series(ax, all_data, "POPC", 30.0, extrusions_list, key, "royalblue")
    add_series(ax, all_data, "DMPC", 50.0, extrusions_list, key, "firebrick")

    ax.set_xlabel("Number of Extrusions")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Comparison of POPC (30°C) and DMPC (50°C)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / f"Comparison_{filename}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    extrusions = [3, 5, 10, 15, 20, 31, 41, 51, 61]
    
    # 1. Diameter Plot
    generate_comparison(extrusions, "peak_size_nm", "Peak Diameter (nm)", "Diameter")
    
    # 2. Width (Sigma) Plot
    generate_comparison(extrusions, "peak_sigma_nm", "Peak Width (nm)", "Width")