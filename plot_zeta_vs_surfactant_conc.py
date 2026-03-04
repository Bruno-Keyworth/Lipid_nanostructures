# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 22:39:39 2026

@author: David Mawson

modified code written by Bruno Keyworth in plot_zeta.py
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER


import matplotlib.colors as mcolors
from itertools import cycle
import cmcrameri.cm as cmc

#==============================================================================
# pick a colormap
cmap = cmc.hawaii.resampled(10)


# generate reversed list of colours from the colormap
colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)][::-1]

# set as the default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Define linestyles and markers
linestyles = cycle(['-', '--', '-.', ':'])
markers = cycle(['o', 's', 'v', 'D', '^', '*', 'x', 'P'])

#==============================================================================

df = pd.read_csv(
    DATA_FOLDER / 'surfactant' / "surfactant_zeta.txt",
    sep="\t",
    skiprows=[1]   # skip units row
)

df.columns = df.columns.str.strip()

df["Measurement Date and Time"] = pd.to_datetime(
    df["Measurement Date and Time"],
    dayfirst=True,
    errors="coerce"
)

df = (
    df.sort_values("Measurement Date and Time")
      .drop_duplicates(subset="Sample Name", keep="last")
)

# Keep required columns
df = df[["Sample Name", "ZP"]].dropna()
df["ZP"] = df["ZP"].astype(float)

# ----------------------------
# Extract surfactant name
# ----------------------------
def extract_surfactant(name):
    m = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", name)
    return m.group(1) if m else "Unknown"

def extract_ratio(name):
    m = re.search(r"\d+\s+POPC\s*:\s*\d+\s+POPG|\d+\s+DMPC\s*:\s*\d+\s+DMPG", name)
    return m.group(0) if m else "Unknown"

df["surfactant"] = df["Sample Name"].apply(extract_surfactant)
df["ratio"] = df["Sample Name"].apply(extract_ratio)

# ----------------------------
# Extract micro-molar concentration
# ----------------------------
def extract_microM(name):
    m = re.search(r"(\d+)\s*microM", name)
    return float(m.group(1)) if m else np.nan


def extract_lipid_ratio(name):
    m = re.search(r"(\d+)\s+([A-Z]+)\s*:\s*(\d+)\s+([A-Z]+)", name)
    if m:
        return {
            "lipid1_n": int(m.group(1)),
            "lipid1": m.group(2),
            "lipid2_n": int(m.group(3)),
            "lipid2": m.group(4),
        }
    return None

def extract_lipid_conc(name):
    m = re.search(r"([\d\.]+)\s*mg_ml", name)
    return float(m.group(1)) if m else np.nan


df["conc_microM"] = df["Sample Name"].apply(extract_microM)

# Drop failed rows
df = df.dropna(subset=["conc_microM", "ZP"])

# ----------------------------
# Group repeats: mean & std per (surfactant, µM)
# ----------------------------
stats = (
    df.groupby(["surfactant", "ratio", "conc_microM"])["ZP"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("conc_microM")
)

# ----------------------------
# Plot: x-axis = microM
# Lines = surfactant type and lipid structure
# ----------------------------

plt.figure(figsize=(11, 5))

for (surf, ratio), sub in stats.groupby(["surfactant", "ratio"]):
    
    ls = next(linestyles)
    mk = next(markers)
    
    sub = sub.sort_values("conc_microM")
    label = f"{surf} | {ratio}"
    
    plt.errorbar(
        sub["conc_microM"],
        sub["mean"],
        yerr=sub["std"],
        marker=mk, linestyle = ls, markeredgecolor="black", 
        markeredgewidth = 0.5, capsize=4,
        label=label
    )

plt.xlabel("Surfactant concentration [µM]")
plt.ylabel("Zeta potential [mV]")
plt.legend(title = "Surfactant | uncharged : charged lipid",fontsize=10, 
           bbox_to_anchor = (1.22,0.5), loc="center")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "ZETA_vs_surfactant_conc.png", dpi=300)
plt.show()


#==============================================================================

