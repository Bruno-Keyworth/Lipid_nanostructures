# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 22:39:39 2026

@author: David
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

# ----------------------------
# Read the data file
# ----------------------------

df = pd.read_csv(
    DATA_FOLDER / 'surfactant' / "surfactant_zeta.txt",
    sep="\t",
    skiprows=[1]   # skip units row
)

df.columns = df.columns.str.strip()

# Keep required columns
df = df[["Sample Name", "ZP"]].dropna()
df["ZP"] = df["ZP"].astype(float)

# ----------------------------
# Extract surfactant name
# ----------------------------
def extract_surfactant(name):
    # assumes surfactant name appears before concentration
    m = re.search(r"(C\d+E\d+)", name)
    return m.group(1) if m else "Unknown"

df["surfactant"] = df["Sample Name"].apply(extract_surfactant)

# ----------------------------
# Extract micro-molar concentration
# ----------------------------
def extract_microM(name):
    m = re.search(r"(\d+)\s*microM", name)
    return float(m.group(1)) if m else np.nan

df["conc_microM"] = df["Sample Name"].apply(extract_microM)

# Drop failed rows
df = df.dropna(subset=["conc_microM", "ZP"])

# ----------------------------
# Group repeats: mean & std per (surfactant, µM)
# ----------------------------
stats = (
    df.groupby(["surfactant", "conc_microM"])["ZP"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("conc_microM")
)

# ----------------------------
# Plot: x-axis = microM
# Lines = surfactant type
# ----------------------------
plt.figure(figsize=(6, 4))

for surf, sub in stats.groupby("surfactant"):
    sub = sub.sort_values("conc_microM")
    plt.errorbar(
        sub["conc_microM"],
        sub["mean"],
        yerr=sub["std"],
        fmt="o-",
        capsize=4,
        label=surf
    )

plt.xlabel("C12E6 concentration (µM)")
plt.ylabel("Zeta potential (mV)")
plt.legend(title="Surfactant")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "zeta_vs_microM.png", dpi=300)
plt.show()