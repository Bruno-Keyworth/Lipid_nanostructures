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

    df = df[np.isclose(df["fraction_DMPG"], ratio)]

    peak_plots = {
        "pos": "Peak Position (nm)",
        "width": "Peak Width (nm)",
        "area": "Peak Area (%)",
    }

    surfactants = ["C12E6", "DDAC", "TX100"]

    control = df[
        (df["C12E6"] == 0) &
        (df["DDAC"] == 0) &
        (df["TX100"] == 0)
    ]

    # ----- Peak plots (one figure per surfactant) -----
    for surf in surfactants:

        sub = pd.concat([control, df[df[surf] > 0]], ignore_index=True)
        sub = sub.sort_values(surf)

        if sub.empty:
            continue

        for key, ylabel in peak_plots.items():

            fig, ax = plt.subplots()

            for peak in [1, 2, 3]:

                val = f"p{peak}_{key}"
                err = f"{val}_err"

                ax.errorbar(
                    sub[surf],
                    sub[val],
                    yerr=sub[err],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=f"Peak {peak}",
                )

            ax.set_xlabel("Surfactant concentration (µM)")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                PLOTS_FOLDER / f"{surf}_{key}_conc.png",
                dpi=300
            )

            plt.show()

    # ----- Zeta plot (all surfactants together) -----
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

def create_fraction_plot(df):

    df["condition"] = df.apply(surfactant_condition, axis=1)
    df = df.dropna(subset=["condition"])

    conditions = ["Control", "C12E6", "DDAC", "TX100"]

    peak_plots = {
        "pos": "Peak Position (nm)",
        "width": "Peak Width (nm)",
        "area": "Peak Area (%)",
    }

    # ----- Peak plots (one figure per surfactant condition) -----
    for cond in conditions:

        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")

        if len(sub) == 0:
            continue

        for key, ylabel in peak_plots.items():

            plt.figure()

            for peak in [1, 2, 3]:

                val = f"p{peak}_{key}"
                err = f"{val}_err"

                plt.errorbar(
                    sub["fraction_DMPG"],
                    sub[val],
                    yerr=sub[err],
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label=f"Peak {peak}",
                )

            plt.xlabel("DMPG Fraction")
            plt.ylabel(ylabel)
            plt.title(cond)
            plt.legend()
            plt.tight_layout()

            safe_cond = cond.replace(" ", "_").replace("µ", "u")

            plt.savefig(
                PLOTS_FOLDER / f"{safe_cond}_{key}_fraction.png",
                dpi=300
            )

            plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ----- absolute zeta -----
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("fraction_DMPG")
        if sub.empty:
            continue
    
        axes[0].errorbar(
            sub["fraction_DMPG"], sub["zeta"], yerr=sub["zeta_err"],
            marker="o", linestyle="-", capsize=3,
            label=cond, color=colours[cond]
        )
    
    # ----- control reference -----
    control = (
        df[df["condition"] == "Control"]
        [["fraction_DMPG", "zeta", "zeta_err"]]
        .set_index("fraction_DMPG")
    )
    
    # ----- control-subtracted zeta -----
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
            merged["fraction_DMPG"], delta, yerr=delta_err,
            marker="o", linestyle="-", capsize=3,
            label=cond, color=colours[cond]
        )
    
    # control reference line
    axes[1].axhline(0, linestyle="--", linewidth=1, color=colours["Control"])
    
    for ax in axes:
        ax.set_xlabel("Fraction DMPG / (DMPC + DMPG)")
        ax.legend()
    
    axes[0].set_ylabel("Zeta Potential (mV)")
    axes[1].set_ylabel("Δ Zeta Potential (mV) relative to control")
    
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "zeta_fraction.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plt.close('all')

    folder = DATA_FOLDER / "surfactants" / "50_degrees"

    df = gather_data(folder)

    create_fraction_plot(df)
    
    folder = DATA_FOLDER / "surfactants" / "25_degrees"

    df = gather_data(folder)
    
    create_concentration_plots(df)