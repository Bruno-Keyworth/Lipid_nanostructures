#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 22:11:13 2026

@author: brunokeyworth
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

# ----------------------------
# Functions
# ----------------------------

def load_zeta_json(folder: Path) -> pd.DataFrame:
    """
    Loads all JSON files in a folder containing zeta measurements.
    Expects each JSON to be a list of dicts with 'sample_name' and 'zeta_mV'.
    """
    rows = []
    for file_path in folder.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        for entry in data:
            # Only keep zeta entries
            if entry.get("type") == "zeta":
                rows.append({
                    "sample_name": entry.get("sample_name"),
                    "ZP": entry.get("zeta_mV")
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No zeta entries found in {folder}")
    return df

def extract_popg_fraction(name: str) -> float:
    m = re.search(r"(\d+)\s*POPC\s*:\s*(\d+)\s*POPG", name)
    if m:
        popc, popg = int(m.group(1)), int(m.group(2))
        return popg / (popc + popg)
    return np.nan

def extract_concentration(name: str) -> float:
    m = re.search(r"([\d.]+)\s*mg_ml", name)
    return float(m.group(1)) if m else np.nan

def clean_zeta_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds POPG fraction and concentration columns, drops rows with missing values.
    """
    df["POPG_fraction"] = df["sample_name"].apply(extract_popg_fraction)
    df["conc_mg_ml"] = df["sample_name"].apply(extract_concentration)
    df["conc_label"] = df["conc_mg_ml"].fillna("unspecified")
    df = df.dropna(subset=["POPG_fraction", "ZP"])
    return df

def aggregate_zeta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by POPG fraction and concentration label, returns mean and std of ZP.
    """
    stats = (
        df.groupby(["POPG_fraction", "conc_label"])["ZP"]
          .agg(["mean", "std"])
          .reset_index()
          .sort_values("POPG_fraction")
    )
    return stats

def plot_zeta_vs_fraction(stats: pd.DataFrame, save_path: Path):
    """
    Plots zeta potential vs POPG fraction with error bars, colored by concentration.
    """
    plt.figure(figsize=(6, 4))

    for conc, sub in stats.groupby("conc_label"):
        plt.errorbar(
            sub["POPG_fraction"],
            sub["mean"],
            yerr=sub["std"],
            fmt="o", ls="-",
            capsize=4,
            label=str(conc)
        )

    plt.xlabel("POPG fraction")
    plt.ylabel("Zeta potential (mV)")
    plt.legend(title="Concentration (mg/ml)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ----------------------------
# Usage
# ----------------------------

folder = DATA_FOLDER / "POPC-POPG"
df = load_zeta_json(folder)
df_clean = clean_zeta_dataframe(df)
stats = aggregate_zeta(df_clean)
plot_zeta_vs_fraction(stats, PLOTS_FOLDER / "ZETA_vs_charged_lipid_fraction.png")