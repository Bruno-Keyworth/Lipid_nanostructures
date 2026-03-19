#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 21:51:53 2026

@author: brunokeyworth
"""

import re

def read_sample_name(base_name):

    lipid_ratio = {"DMPC":0, "DMPG":0, "POPC":0, "POPG":0, "DPPC":0,}
    surfactant_concentration = {"C12E6":0, "DDAC":0, "TX100":0}

    extrusions = 31
    lipid_concentration_mg_ml = 0.2

    # -------- lipid ratio --------
    ratio_match = re.search(r"(\d+)\s*(DMPC|POPC)\s*:\s*(\d+)\s*(DMPG|POPG)", base_name)
    if ratio_match:
        n1, l1, n2, l2 = ratio_match.groups()
        lipid_ratio[l1] = int(n1)
        lipid_ratio[l2] = int(n2)
    else:
        # -------- single lipid case --------
        single_match = re.search(r"\b(DMPC|DMPG|POPC|POPG|DPPC)\b", base_name)
        if single_match:
            lipid_ratio[single_match.group(1)] = 10

    # -------- extrusion number --------
    extr_match = re.search(r"(\d+)\s+Extrusion", base_name)
    if extr_match:
        extrusions = int(extr_match.group(1))

    # -------- lipid concentration --------
    conc_match = re.search(r"([\d\.]+)\s*mg_ml", base_name)
    if conc_match:
        lipid_concentration_mg_ml = float(conc_match.group(1))

    # -------- surfactant concentration --------
    surf_match = re.search(r"(C12E6|DDAC|TX100)\s+(\d+)\s*microM", base_name)
    if surf_match:
        surf, conc = surf_match.groups()
        surfactant_concentration[surf] = int(conc)

    return {
        "base_sample_name": base_name,
        "lipid_ratio": lipid_ratio,
        "surfactant_conc_microM": surfactant_concentration,
        "extrusions": extrusions,
        "lipid_conc_mg_ml": lipid_concentration_mg_ml
    }
