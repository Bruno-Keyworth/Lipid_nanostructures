# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:45:00 2026

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER

import matplotlib.colors as mcolors
from itertools import cycle
import cmcrameri.cm as cmc

def colours_style(length):
    cmap = cmc.hawaii.resampled(len(unique_combos))

    # generate reversed list of colours from the colormap
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)][::-1]

    # set as the default color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # Define linestyles and markers
    linestyles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['o', 's', 'v', 'D', '^', '*', 'x', 'P', 'h', '+', '.', '>'])
    
    return markers, linestyles


filepath = DATA_FOLDER / 'surfactant' / "surfactant_zeta.txt"

with open(filepath, "r") as f:
    lines = f.readlines()

# Remove header + units row
data_lines = lines[2:]


# Containers
zeta = []
surfactant = []
conc_microM = []
charged_fraction = []
ratio_label = []


for line in data_lines:

    parts = line.strip().split("\t")

    if len(parts) < 6:
        continue

    sample_name = parts[2]
    zp_value = parts[5]

    if not zp_value.replace(".", "", 1).replace("-", "", 1).isdigit():
        continue

    zeta.append(float(zp_value))

    # surfactant 
    m = re.search(r"\b(C\d+E\d+|DDAC|TX100|NONE)\b", sample_name)
    surfactant.append(m.group(1) if m else "Unknown")

    #  concentration 
    m = re.search(r"(\d+)\s*microM", sample_name)
    conc_microM.append(float(m.group(1)) if m else np.nan)

    # Lipid ratio
    m = re.search(r"(\d+)\s+([A-Z]+)\s*:\s*(\d+)\s+([A-Z]+)", sample_name)
    if m:
        n1 = int(m.group(1))
        l1 = m.group(2)
        n2 = int(m.group(3))
        l2 = m.group(4)

        frac = n2 / (n1 + n2)
        charged_fraction.append(frac)

        ratio_label.append(f"{n1}:{n2} {l1}:{l2}")
    else:
        charged_fraction.append(np.nan)
        ratio_label.append("Unknown")

# Convert to arrays
zeta = np.array(zeta)
surfactant = np.array(surfactant)
conc_microM = np.array(conc_microM)
charged_fraction = np.array(charged_fraction)
ratio_label = np.array(ratio_label)

# Remove NaNs
mask = ~np.isnan(conc_microM) & ~np.isnan(charged_fraction)
zeta = zeta[mask]
surfactant = surfactant[mask]
conc_microM = conc_microM[mask]
charged_fraction = charged_fraction[mask]
ratio_label = ratio_label[mask]










#==============================================================================
# plot ZETA vs surfactant conc
# GROUP BY (surfactant, ratio)


unique_combos = np.unique(np.stack((surfactant, ratio_label), axis=1),axis=0)

markers, linestyles = colours_style(len(unique_combos))


plt.figure(figsize=(11,6))

for surf, ratio in unique_combos:

    mask_combo = (surfactant == surf) & (ratio_label == ratio)

    combo_zeta = zeta[mask_combo]
    combo_conc = conc_microM[mask_combo]

    unique_conc = np.unique(combo_conc)

    means = []
    stds = []

    for c in unique_conc:
        mask_c = combo_conc == c
        means.append(np.mean(combo_zeta[mask_c]))
        stds.append(np.std(combo_zeta[mask_c], ddof=1))

    unique_conc = np.array(unique_conc)
    means = np.array(means)
    stds = np.array(stds)

    order = np.argsort(unique_conc)
    
    ls = next(linestyles)
    mk = next(markers)
    
    plt.errorbar(
        unique_conc[order],
        means[order],
        yerr=stds[order],
        marker=mk, linestyle = ls, markeredgecolor="black", 
        markeredgewidth = 0.5, capsize=4,
        label=f"{surf} | {ratio}"
    )

# ==========================================================
# PLOT


plt.xlabel("Surfactant concentration (µM)")
plt.ylabel("Zeta potential (mV)")
plt.title("Zeta potential vs surfactant concentration")
plt.legend(bbox_to_anchor=(1.22, 0.5), loc="center")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "surfactant_ZETA_vs_lipid_fraction.png", dpi=300)
plt.show()





# ==========================================================
#FIX SURFACTANT CONCENTRATION

fixed_conc = 100  # change as needed

mask_conc = conc_microM == fixed_conc

zeta = zeta[mask_conc]
surfactant = surfactant[mask_conc]
charged_fraction = charged_fraction[mask_conc]

unique_surf = np.unique(surfactant)

markers, linestyles = colours_style(len(unique_surf))


plt.figure(figsize=(8,5))

for surf in unique_surf:

    mask_s = surfactant == surf

    surf_zeta = zeta[mask_s]
    surf_frac = charged_fraction[mask_s]

    unique_frac = np.unique(surf_frac)

    means = []
    stds = []

    for f in unique_frac:
        mask_f = surf_frac == f
        means.append(np.mean(surf_zeta[mask_f]))
        stds.append(np.std(surf_zeta[mask_f], ddof=1))

    unique_frac = np.array(unique_frac)
    means = np.array(means)
    stds = np.array(stds)

    # Sort by fraction
    order = np.argsort(unique_frac)

    ls = next(linestyles)
    mk = next(markers)
    
    plt.errorbar(
        unique_frac[order],
        means[order],
        yerr=stds[order],
        marker=mk, linestyle = ls, markeredgecolor="black", 
        markeredgewidth = 0.5, capsize=4,
        label=surf
    )

# ==========================================================
# PLOT

plt.xlabel("Charged lipid fraction")
plt.ylabel("Zeta potential (mV)")
plt.title(f"Zeta vs charged fraction ({fixed_conc} µM)")
plt.legend(title="Surfactant")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "ZETA_vs_surfactant_conc.png", dpi=300)
plt.show()