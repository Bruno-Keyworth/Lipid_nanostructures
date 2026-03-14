#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:12:38 2026

@author: brunokeyworth
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors

from get_filepaths import DATA_FOLDER, PLOTS_FOLDER


def charged_fraction(lipid_ratio):

    total = sum(lipid_ratio.values())
    if total == 0:
        return None

    charged = lipid_ratio.get("DMPG", 0) + lipid_ratio.get("POPG", 0)
    return charged / total


def extract_surfactant_and_conc(surf_dict):

    for surf, conc in surf_dict.items():
        if conc > 0:
            return surf, conc

    return "NONE", 0.0


def load_results():

    folder = DATA_FOLDER / "surfactants" / "50_degrees"

    rows = []

    for file_path in folder.glob("*.json"):

        with open(file_path) as f:
            entry = json.load(f)

        peaks = entry.get("averaged_peaks", [])
        if not peaks:
            continue

        # dominant peak by area
        peak = max(peaks, key=lambda p: p.get("area_percent", [0])[0])

        surf, conc = extract_surfactant_and_conc(
            entry.get("surfactant_concentration", {})
        )

        rows.append({
            "charged_fraction": charged_fraction(entry.get("lipid_ratio", {})),
            "surfactant": surf,
            "conc_microM": conc,
            "peak_nm": peak.get("peak_position_nm", [None])[0],
            "sigma_nm": peak.get("peak_width_nm", [None])[0],
        })

    return pd.DataFrame(rows)


def plot_peak_vs_fraction(df, fixed_conc=100):

    df = df[df["conc_microM"].apply(lambda x: np.isclose(x, fixed_conc) or np.isclose(x, 0))]

    groups = df["surfactant"].unique()

    cmap = cmc.hawaii.resampled(len(groups))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, surf in enumerate(groups):

        sub = df[df["surfactant"] == surf].sort_values("charged_fraction")

        ax[0].errorbar(
            sub["charged_fraction"], sub["peak_nm"], yerr=0,
            marker="o", color=colors[i], label=surf, ls=''
        )

        ax[1].errorbar(
            sub["charged_fraction"], sub["sigma_nm"], yerr=0,
            marker="o", color=colors[i], label=surf, ls=''
        )

    for axes in ax:
        axes.set_xlabel("Charged lipid fraction")
        axes.set_title(f"50 °C, surfactant = {fixed_conc} µM")
        axes.set_yscale("log")
        axes.legend()

    ax[0].set_ylabel("Mean peak size (nm)")
    ax[1].set_ylabel("Mean peak width σ (nm)")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "size_vs_lipid_fraction_50C.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    df = load_results()

    plot_peak_vs_fraction(df)
