# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:13:22 2026

@author: David Mawson

modified version of code written by bruno keyworth
"""


import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from get_filepaths import DATA_FOLDER, PLOTS_FOLDER
import re

aging_FOLDER = DATA_FOLDER / "aging"
extrusion = 31
temperatures = [10, 20, 30, 40, 50, 60]
time_format = "%d %B %Y %H:%M:%S"
LIPIDS = ["7 DMPC : 3 DMPG", "DMPC", "DMPG", "POPC", "DPPC"]

def load_entries():
    entries = []
    for file in aging_FOLDER.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data:
            entry = standardise_sample(d)
            if entry is None:
                continue

            entries.append(entry)

    return entries


def extract_peak_diameter(entry):
    """Take diameter from peak with largest area."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None
    peak = max(peaks, key=lambda x: float(x.get("area_percent") or 0))
    try:
        return float(peak.get("peak_position_nm") or 0)
    except Exception:
        return None


def extract_peak_width(entry):
    """Take width from peak with largest area."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None
    peak = max(peaks, key=lambda x: float(x.get("area_percent") or 0))
    try:
        return float(peak.get("peak_width_nm") or 0)
    except Exception:
        return None

def standardise_sample(entry):
    sample_name = entry.get("sample_name", "")
    if not sample_name:
        return None
    
    #filter not 31 extrusions
    if "Extrusion" in sample_name:
        
        if not sample_name.startswith("31 Extrusion"):
            return None  # reject

    # add lipid name
    for lipid in LIPIDS:
        if lipid in sample_name:
            entry["lipid"] = lipid
            return entry

    return None  # reject if no lipid match

# ----------------------------
# load data
# ----------------------------
all_entries = load_entries()



def find_lipid_start(entries):
    dates = []
    for entry in entries:
        ts = entry.get("timestamp")
        if ts:
            try:
                dt = datetime.strptime(ts, time_format).date()
                dates.append(dt)
            except Exception:
                continue
    if not dates:
        raise RuntimeError("No valid timestamps found.")
    return min(dates)


def time_since(start, end):
    time = datetime.strptime(end, time_format).date()
    return (time - start).days

def plot_line(ax, name, entries, extractor):
    
    values_by_day = defaultdict(list)
    
    # Group values by day
    for entry in entries:
        
        day = time_since(entry['lipid_start_time'], entry['timestamp'])
        
        value = extractor(entry)
        #print(name, entry['lipid_start_time'], entry['timestamp'], value)
        if value is None:
            continue
        
        values_by_day[day].append(value)
    
    # Compute averages
    days_sorted = sorted(values_by_day.keys())
    avg_values = [np.mean(values_by_day[day]) for day in days_sorted]
    std_values = [np.std(values_by_day[day]) for day in days_sorted]
    
    ax.errorbar(days_sorted, avg_values, yerr=std_values, ls ="-",fmt='o', label=f"{name}", capsize=4)
    return None

def plot_aging_lipids(entries):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    entries_by_lipid = defaultdict(list)
    
    for entry in entries:
        lipid = entry.get("lipid")
        if lipid:
            entries_by_lipid[lipid].append(entry)
    
    for lipid, lipid_entries in entries_by_lipid.items():
        
        lipid_start_time = find_lipid_start(lipid_entries)
        
        for entry in lipid_entries:
            entry["lipid_start_time"] = lipid_start_time
    
    for lipid, lipid_entries in entries_by_lipid.items():
        plot_line(ax1, lipid, lipid_entries, extractor=extract_peak_diameter)
        plot_line(ax2, lipid, lipid_entries, extractor=extract_peak_width)

        

    ax1.set_xlabel("Time since extrusion (days)")
    ax2.set_xlabel("Time since extrusion (days)")
    ax1.set_ylabel("Peak Diameter (nm)")
    ax2.set_ylabel("Peak Width (nm)")
    ax1.legend()
    ax2.legend()
    ax1.grid(linestyle="--", alpha=0.3)
    ax2.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "time_since_extrusion_all_lipids.png", dpi=300)
    plt.show()
    return None


plot_aging_lipids( all_entries)
