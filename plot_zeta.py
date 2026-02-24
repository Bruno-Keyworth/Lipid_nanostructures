#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 22:11:13 2026

@author: brunokeyworth
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

# ----------------------------
# Read the data file
# ----------------------------

df = pd.read_csv(
    DATA_FOLDER / 'POPC-POPG' / "zeta.txt",
    sep="\t",
    skiprows=[1]  # skip units row
)
# Clean column names
df.columns = df.columns.str.strip()

# Keep only required columns
df = df[["Sample Name", "ZP"]].dropna()
df["ZP"] = df["ZP"].astype(float)


# ----------------------------

# ----------------------------
# Extract POPG fraction
# ----------------------------
def extract_popg_fraction(name):
    m = re.search(r"(\d+)\s*POPC\s*:\s*(\d+)\s*POPG", name)
    if m:
        popc = int(m.group(1))
        popg = int(m.group(2))
        return popg / (popc + popg)
    return np.nan

df["POPG_fraction"] = df["Sample Name"].apply(extract_popg_fraction)

# Drop rows where parsing failed
df = df.dropna(subset=["POPG_fraction"])

# ----------------------------
# Group repeats: mean & std
# ----------------------------
stats = (
    df.groupby("POPG_fraction")["ZP"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("POPG_fraction")
)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6, 4))
plt.errorbar(
    stats["POPG_fraction"],
    stats["mean"],
    yerr=stats["std"],
    fmt="o",
    capsize=4
)
plt.xlabel("POPG fraction")
plt.ylabel("Zeta potential (mV)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / 'zeta.png', dpi=300)
plt.show()