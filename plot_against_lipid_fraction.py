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

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

colours = {
    "Control": "black",
    "C12E6": "tab:blue",
    "DDAC": "tab:red",
    "TX100": "tab:green",
}

def mean_err(entry):
    if entry and isinstance(entry, list) and len(entry) == 2:
        return entry[0], entry[1]
    return np.nan, np.nan


def gather_data(folder):

    records = []

    for fp in folder.glob("*.json"):
        with open(fp) as f:
            data = json.load(f)

        if not np.isclose(data.get("lipid_conc_mg_ml", np.nan), 0.1):
            continue

        lipid = data["lipid_ratio"]
        surf = data["surfactant_conc_microM"]

        DMPC = lipid.get("DMPC", 0)
        DMPG = lipid.get("DMPG", 0)

        if DMPC == 0 and DMPG == 0:
            continue

        total = DMPC + DMPG
        frac_DMPG = DMPG / total if total > 0 else np.nan

        surf_total = surf.get("C12E6", 0) + surf.get("DDAC", 0) + surf.get("TX100", 0)

        peaks = data.get("averaged_peaks", [])

        p1 = peaks[0] if len(peaks) > 0 else {}
        p2 = peaks[1] if len(peaks) > 1 else {}
        p3 = peaks[2] if len(peaks) > 2 else {}

        zeta, zeta_err = mean_err(data.get("average_zeta"))

        p1_pos, p1_pos_err = mean_err(p1.get("peak_position_nm"))
        p1_width, p1_width_err = mean_err(p1.get("peak_width_nm"))
        p1_area, p1_area_err = mean_err(p1.get("area_percent"))

        p2_pos, p2_pos_err = mean_err(p2.get("peak_position_nm"))
        p2_width, p2_width_err = mean_err(p2.get("peak_width_nm"))
        p2_area, p2_area_err = mean_err(p2.get("area_percent"))

        p3_pos, p3_pos_err = mean_err(p3.get("peak_position_nm"))
        p3_width, p3_width_err = mean_err(p3.get("peak_width_nm"))
        p3_area, p3_area_err = mean_err(p3.get("area_percent"))

        record = {
            "fraction_DMPG": frac_DMPG,
            "surfactant_total": surf_total,

            "zeta": zeta,
            "zeta_err": zeta_err,

            "p1_pos": p1_pos,
            "p1_pos_err": p1_pos_err,
            "p1_width": p1_width,
            "p1_width_err": p1_width_err,
            "p1_area": p1_area,
            "p1_area_err": p1_area_err,

            "p2_pos": p2_pos,
            "p2_pos_err": p2_pos_err,
            "p2_width": p2_width,
            "p2_width_err": p2_width_err,
            "p2_area": p2_area,
            "p2_area_err": p2_area_err,

            "p3_pos": p3_pos,
            "p3_pos_err": p3_pos_err,
            "p3_width": p3_width,
            "p3_width_err": p3_width_err,
            "p3_area": p3_area,
            "p3_area_err": p3_area_err,

            "C12E6": surf.get("C12E6", 0),
            "DDAC": surf.get("DDAC", 0),
            "TX100": surf.get("TX100", 0),
        }

        records.append(record)

    return pd.DataFrame(records)

def surfactant_condition(row):

    if row["C12E6"] == 0 and row["DDAC"] == 0 and row["TX100"] == 0:
        return "Control"

    if row["C12E6"] == 100:
        return "C12E6"
    if row["DDAC"] == 100:
        return "DDAC"
    if row["TX100"] == 100:
        return "TX100"

    return None

def create_concentration_plots(df, ratio=0.3):
    """
    Creates:
    - Peak plots (position, width, area) as 2x2 grid (Control + 3 surfactants)
    - Zeta potential plot (single plot for all surfactants)
    """
    df = df[np.isclose(df["fraction_DMPG"], ratio)]

    peak_plots = {
        "pos": "Peak Position (nm)",
        "width": "Peak Width (nm)",
        "area": "Peak Area (%)",
    }

    surfactants = ["C12E6", "DDAC", "TX100"]

    # Identify control rows
    control = df[(df["C12E6"] == 0) & (df["DDAC"] == 0) & (df["TX100"] == 0)]

    # ----- Peak plots (2x2 grid for control + each surfactant) -----
    conditions = ["Control"] + surfactants
    for key, ylabel in peak_plots.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, cond in enumerate(conditions):
            if cond == "Control":
                sub = control
                x = np.zeros(len(sub))  # Control points at 0 µM
            else:
                sub = pd.concat([control, df[df[cond] > 0]], ignore_index=True)
                sub = sub.sort_values(cond)
                x = sub[cond]

            if sub.empty:
                continue

            ax = axes[i]
            for peak in [1, 2, 3]:
                val = f"p{peak}_{key}"
                err = f"{val}_err"
                ax.errorbar(
                    x,
                    sub[val],
                    yerr=sub[err],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=f"Peak {peak}",
                )

            ax.set_title(cond)
            ax.set_xlabel("Surfactant concentration (µM)")
            ax.set_xscale("log" if cond != "Control" else "linear")
            ax.set_ylabel(ylabel)
            ax.legend()

        # Remove unused subplots if less than 4
        for j in range(len(conditions), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.savefig(PLOTS_FOLDER / f"{key}_conc_2x2.png", dpi=300)
        plt.show()

    # ----- Zeta plot (single plot for all surfactants) -----
    fig, ax = plt.subplots()
    for surf in surfactants:
        sub = pd.concat([control, df[df[surf] > 0]], ignore_index=True)
        sub = sub.sort_values(surf)
        if sub.empty:
            continue

        ax.errorbar(
            sub[surf],
            sub["zeta"],
            yerr=sub["zeta_err"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=surf,
            color=colours[surf],
        )

    ax.set_xlabel("Surfactant concentration (µM)")
    ax.set_ylabel("Zeta Potential (mV)")
    ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_conc.png", dpi=300)
    plt.show()

def create_fraction_plots(df):
    """
    Creates:
    - Peak plots (position, width, area) as 2x2 grid (Control + 3 surfactants)
    - Zeta potential plots: absolute and delta vs control
    """
    # Add the condition column based on surfactant concentrations
    df["condition"] = df.apply(surfactant_condition, axis=1)
    df = df.dropna(subset=["condition"])  # remove any rows without a condition

    # Now you can safely use df["condition"]
    conditions = ["Control", "C12E6", "DDAC", "TX100"]

    ...
    
    # Peak properties
    peak_plots = {
        "pos": "Peak Position (nm)",
        "width": "Peak Width (nm)",
        "area": "Peak Area (%)",
    }

    # ----- Peak plots -----
    for key, ylabel in peak_plots.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()  # easier indexing

        for i, cond in enumerate(conditions):
            sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
            if sub.empty:
                continue

            ax = axes[i]
            for peak in [1, 2, 3]:
                val = f"p{peak}_{key}"
                err = f"{val}_err"
                ax.errorbar(
                    sub["fraction_DMPG"],
                    sub[val],
                    yerr=sub[err],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=f"Peak {peak}",
                )
            ax.set_title(cond)
            ax.set_xlabel("DMPG Fraction")
            ax.set_ylabel(ylabel)
            ax.legend()

        # Remove empty subplots if less than 4 conditions
        for j in range(len(conditions), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.savefig(PLOTS_FOLDER / f"{key}_fraction_2x2.png", dpi=300)
        plt.show()

    # ----- Zeta plots (absolute + delta relative to control) -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Absolute zeta
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        if sub.empty:
            continue
        axes[0].errorbar(
            sub["fraction_DMPG"],
            sub["zeta"],
            yerr=sub["zeta_err"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=cond,
            color=colours[cond],
        )

    # Control reference
    control = df[df["condition"] == "Control"][["fraction_DMPG", "zeta", "zeta_err"]].set_index("fraction_DMPG")

    # Control-subtracted zeta
    for cond in conditions[1:]:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        merged = sub.merge(
            control,
            left_on="fraction_DMPG",
            right_index=True,
            suffixes=("_surf", "_ctrl"),
        )
        if merged.empty:
            continue
        delta = merged["zeta_surf"] - merged["zeta_ctrl"]
        delta_err = np.sqrt(merged["zeta_err_surf"]**2 + merged["zeta_err_ctrl"]**2)

        axes[1].errorbar(
            merged["fraction_DMPG"],
            delta,
            yerr=delta_err,
            marker="o",
            linestyle="-",
            capsize=3,
            label=cond,
            color=colours[cond],
        )

    # Reference line at 0
    axes[1].axhline(0, linestyle="--", linewidth=1, color=colours["Control"])

    for ax in axes:
        ax.set_xlabel("Fraction DMPG / (DMPC + DMPG)")
        ax.legend()

    axes[0].set_ylabel("Zeta Potential (mV)")
    axes[1].set_ylabel("Δ Zeta Potential (mV) relative to control")

    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / "zeta_fraction.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plt.close('all')

    folder = DATA_FOLDER / "surfactants" / "50_degrees"

    df = gather_data(folder)

    create_fraction_plots(df)
    
    folder = DATA_FOLDER / "surfactants" / "25_degrees"

    df = gather_data(folder)
    
    create_concentration_plots(df)