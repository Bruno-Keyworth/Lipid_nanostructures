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

extrusions = [3, 5, 10, 11, 15, 20, 21, 31, 41, 51, 61]

# ----------------------------
# DATA LOADING
# ----------------------------

def load_measurements(parent_folder="extrusions"):
    folder_path = DATA_FOLDER / parent_folder
    processed_entries = []

    for file_path in folder_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                entries = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue

        for entry in entries:
            ratios = entry.get("lipid_ratio", {})

            lipid_type = None
            if ratios.get("POPC") == 10:
                lipid_type = "POPC"
            elif ratios.get("DMPC") == 10:
                lipid_type = "DMPC"
            elif ratios.get("DOPC") == 10:
                lipid_type = "DOPC"

            if not lipid_type:
                continue

            temp = entry.get("temperature_C", 0)
            n_extr = entry.get("extrusions", 0)

            peaks = entry.get("repeat_peaks", [])
            for repeat in peaks:
                if not repeat:
                    continue

                best_peak = max(repeat, key=lambda x: x.get("area_percent", 0))

                processed_entries.append({
                    "lipid": lipid_type,
                    "temperature_C": float(temp),
                    "extrusions": int(n_extr),
                    "peak_size_nm": float(best_peak.get("peak_position_nm", np.nan)),
                    "peak_sigma_nm": float(best_peak.get("peak_width_nm", np.nan))
                })

    return processed_entries


# ----------------------------
# MODEL (POPC / DMPC ONLY)
# ----------------------------

def model(n, D_inf, D0, N):
    return D_inf + (D0 - D_inf) * np.exp(-n / N)

def compute_n_star(D_inf, D0, N, x_data, y_err, epsilon=0.05):
    """
    Data-driven n* using:
    - actual sampled extrusion points (not n+1)
    - propagated measurement uncertainty
    """

    def D(n):
        return D_inf + (D0 - D_inf) * np.exp(-n / N)

    x_data = np.array(x_data)
    y_err = np.array(y_err)

    # sort to ensure monotonic ordering
    order = np.argsort(x_data)
    x = x_data[order]
    err = y_err[order]

    D_vals = D(x)

    # model change between successive measured points
    delta_model = np.abs(np.diff(D_vals))

    # propagated experimental uncertainty
    sigma_delta = np.sqrt(err[:-1]**2 + err[1:]**2)

    # asymptotic condition
    asymptote = np.abs((D_vals[:-1] - D_inf) / D_inf)

    mask = (delta_model < sigma_delta) & (asymptote < epsilon)

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return np.nan

    # return first valid extrusion point
    return x[idx[0]]


# ----------------------------
# FITTING + PLOTTING
# ----------------------------

def add_series(ax, data, lipid, target_temp, extrusions_list, key, color, fit=True):

    means, errors = [], []
    subset = [d for d in data if d["lipid"] == lipid and d["temperature_C"] == target_temp]

    for e in extrusions_list:

        vals = [
            d[key] for d in subset
            if d["extrusions"] == e and np.isfinite(d[key]) and d[key] > 0
        ]

        if len(vals) > 1:
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            err = max(std, 0.15 * mean if e <= 10 else 0.05 * mean)

        elif len(vals) == 1:
            mean = vals[0]
            err = 0.15 * mean if e <= 10 else 0.05 * mean

        else:
            mean = np.nan
            err = np.nan

        means.append(mean)
        errors.append(err)

    means = np.array(means)
    errors = np.array(errors)
    x_data = np.array(extrusions_list)

    mask = (~np.isnan(means)) & (~np.isnan(errors)) & (errors > 0)

    if not np.any(mask):
        return

    # ----------------------------
    # DATA PLOT
    # ----------------------------
    ax.errorbar(
        x_data[mask],
        means[mask],
        yerr=errors[mask],
        fmt='o',
        color=color,
        label=f"{lipid} Data",
        capsize=4
    )

    # ----------------------------
    # FIT ONLY FOR POPC / DMPC
    # ----------------------------
    if not fit:
        return

    try:
        p0 = [
            np.mean(means[mask][-2:]),  # D_inf
            np.mean(means[mask][:2]),   # D0
            15
        ]

        popt, pcov = curve_fit(
            model,
            x_data[mask],
            means[mask],
            p0=p0,
            sigma=errors[mask],
            absolute_sigma=True,
            maxfev=10000
        )

        perr = np.sqrt(np.diag(pcov))

        n_star = compute_n_star(popt[0], popt[1], popt[2], x_data[mask], errors[mask])

        print(f"\n{lipid} at {target_temp}°C, {key}")
        print(f"D_inf = {popt[0]:.3f} ± {perr[0]:.3f}")
        print(f"D0    = {popt[1]:.3f} ± {perr[1]:.3f}")
        print(f"N     = {popt[2]:.3f} ± {perr[2]:.3f}")
        print(f"n*    = {n_star:.1f}")

        x_fit = np.linspace(min(extrusions_list), max(extrusions_list), 100)

        ax.plot(
            x_fit,
            model(x_fit, *popt),
            '--',
            color=color,
            alpha=0.7,
            label=f"{lipid} Fit"
        )
        y_fit = model(x_data[mask], *popt)

        residuals = means[mask] - y_fit
        
        chi2 = np.sum((residuals / errors[mask])**2)
        
        dof = len(y_fit) - len(popt)
        
        chi2_red = chi2 / dof if dof > 0 else np.nan
        
        print(f"Reduced chi^2 = {chi2_red:.3f}")

    except Exception as e:
        print(f"Fit failed for {lipid}: {e}")


# ----------------------------
# FIGURE
# ----------------------------

def generate_double_figure(extrusions_list, key1, key2, filename):

    all_data = (
        load_measurements("extrusions") +
        load_measurements("data_from_kate")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # FITTED SPECIES
    add_series(ax1, all_data, "POPC", 30.0, extrusions_list, key1, "royalblue", fit=True)
    add_series(ax1, all_data, "DMPC", 50.0, extrusions_list, key1, "firebrick", fit=True)

    add_series(ax2, all_data, "POPC", 30.0, extrusions_list, key2, "royalblue", fit=True)
    add_series(ax2, all_data, "DMPC", 50.0, extrusions_list, key2, "firebrick", fit=True)

    # DOPC: DATA ONLY (NO FIT)
    add_series(ax1, all_data, "DOPC", 25.0, extrusions_list, key1, "forestgreen", fit=False)
    add_series(ax2, all_data, "DOPC", 25.0, extrusions_list, key2, "forestgreen", fit=False)

    ax1.set_ylabel("Hydrodynamic Diameter (nm)", fontsize=20)
    ax2.set_ylabel("Peak Width (nm)", fontsize=20)

    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=16)
        ax.set_xlabel("Extrusion Passes", fontsize=20)

    handles, labels = ax1.get_legend_handles_labels()
    
    order = [2, 0, 3, 1, 4]
    
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, 0)
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(PLOTS_FOLDER / f"Comparison_{filename}.png", dpi=300)
    plt.show()


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    generate_double_figure(extrusions, "peak_size_nm", "peak_sigma_nm", "both")