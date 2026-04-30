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

def parse_sample_name(name):
    """
    Returns:
        base_name (without replicate index)
        replicate_id (int or None)
    """
    match = re.match(r"^(.*)\s+(\d+)\s*$", name)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return name.strip(), None
def deduplicate_latest_per_day(entries):
    latest = {}

    for entry in entries:
        sample = entry.get("sample_name")
        ts = entry.get("timestamp")
        start = entry.get("lipid_start_time")

        if not (sample and ts and start):
            continue

        if not entry.get("peaks"):
            continue

        try:
            day = time_since(start, ts)
            dt = datetime.strptime(ts, time_format)
        except Exception:
            continue

        base, rep = parse_sample_name(sample)

        key = (base, rep, day)

        if key not in latest or dt > latest[key]["_dt"]:
            entry["_dt"] = dt
            latest[key] = entry

    for e in latest.values():
        e.pop("_dt", None)

    return list(latest.values())
    
def extract_peak_diameter(entry):
    """Take diameter from smallest-size peak."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None

    try:
        peak = min(peaks, key=lambda x: float(x.get("peak_position_nm") or np.inf))
        if float(peak.get("peak_position_nm")) > 200:
            return np.nan
        return float(peak.get("peak_position_nm") or np.nan)
    except Exception:
        return None


def extract_peak_width(entry):
    """Take width from smallest-size peak."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None

    try:
        peak = min(peaks, key=lambda x: float(x.get("peak_position_nm") or np.inf))
        return float(peak.get("peak_width_nm") or np.nan)
    except Exception:
        return None
    
def extract_all_peaks(entry):
    peaks = entry.get("peaks", [])
    results = []

    for p in peaks:
        try:
            diameter = float(p.get("peak_position_nm") or np.nan)
            width = float(p.get("peak_width_nm") or np.nan)

            if np.isnan(diameter):
                continue

            results.append((diameter, width))
        except Exception:
            continue

    return results

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

def plot_line_split(ax, name, entries, value_index):

    low = defaultdict(list)
    high = defaultdict(list)

    for entry in entries:
        day = time_since(entry['lipid_start_time'], entry['timestamp'])
        peaks = extract_all_peaks(entry)

        for diameter, width in peaks:
            target = low if diameter < 200 else high
            value = diameter if value_index == 0 else width
            target[day].append(value)

    # stable colour assignment
    color = ax._get_lines.get_next_color()

    def plot_group(values_by_day):
        if not values_by_day:
            return None

        days = sorted(values_by_day.keys())
        avg = [np.mean(values_by_day[d]) for d in days]
        std = [np.std(values_by_day[d]) for d in days]

        return ax.errorbar(
            days, avg, yerr=std,
            ls="-", fmt='o',
            color=color,
            capsize=4
        )

    h1 = plot_group(low)
    h2 = plot_group(high)

    if h1 is not None:
        h1[0].set_label(name)

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
            
    for lipid in entries_by_lipid:
        entries_by_lipid[lipid] = deduplicate_latest_per_day(entries_by_lipid[lipid])
    
    for lipid, lipid_entries in entries_by_lipid.items():

        if lipid == "7 DMPC : 3 DMPG":
            plot_line_split(ax1, lipid, lipid_entries, value_index=0)  # diameter
            plot_line_split(ax2, lipid, lipid_entries, value_index=1)  # width
        else:
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
