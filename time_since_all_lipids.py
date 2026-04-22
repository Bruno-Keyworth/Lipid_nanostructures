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

def deduplicate_latest_per_day(entries):
    """
    Keep only the latest entry per (sample_name, day).
    """
    latest = {}

    for entry in entries:
        sample = entry.get("sample_name")
        ts = entry.get("timestamp")
        start = entry.get("lipid_start_time")

        if not (sample and ts and start):
            continue

        try:
            day = time_since(start, ts)
            dt = datetime.strptime(ts, time_format)
        except Exception:
            continue

        key = (sample, day)

        # keep the latest timestamp
        if key not in latest or dt > latest[key]["_dt"]:
            entry["_dt"] = dt  # store parsed datetime temporarily
            latest[key] = entry

    # remove helper field before returning
    for e in latest.values():
        e.pop("_dt", None)

    return list(latest.values())

def keep_last_n_per_day(entries, n):
    grouped = defaultdict(list)

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

        key = (sample, day)
        entry["_dt"] = dt 
        grouped[key].append(entry)

    result = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x["_dt"])
        last_n = group_sorted[-n:]  # take last n
        result.extend(last_n)

    for e in result:
        e.pop("_dt", None)

    return result
    
def extract_peak_diameter(entry, min_peak):
    """Take diameter from smallest-size peak."""
    peaks = entry.get("peaks", [])
    if not peaks:
        return None

    try:
        if min_peak:
            peak = min(peaks, key=lambda x: float(x.get("peak_position_nm") or np.inf))
        else:
            peak = max(peaks, key=lambda x: float(x.get("peak_position_nm") or 0))
        return float(peak.get("peak_position_nm") or np.nan)
    except Exception:
        return None


def extract_peak_width(entry, min_peak):
    """Take width from smallest-size peak."""
    peaks = entry.get("peaks", [])
    if "09 April" in entry.get("timestamp"):
        if "7" in entry.get("lipid"):
            if "size" in entry.get("type"):
                print("====")
    if not peaks:
        return None

    try:
        if min_peak:
            peak = min(peaks, key=lambda x: float(x.get("peak_position_nm") or np.inf))
        else:
            peak = max(peaks, key=lambda x: float(x.get("peak_position_nm") or 0))
        return float(peak.get("peak_width_nm") or np.nan)
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


def plot_line(ax, name, entries, extractor, min_peak = True):
    
    values_by_day = defaultdict(list)
    
    # Group values by day
    for entry in entries:
        
        day = time_since(entry['lipid_start_time'], entry['timestamp'])
        
        value = extractor(entry, min_peak)
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
            
    for lipid in entries_by_lipid:
        #entries_by_lipid[lipid] = deduplicate_latest_per_day(entries_by_lipid[lipid])
        entries_by_lipid[lipid] = keep_last_n_per_day(entries_by_lipid[lipid], n=3)
    
    for lipid, lipid_entries in entries_by_lipid.items():
        plot_line(ax1, lipid, lipid_entries, extractor=extract_peak_diameter, min_peak = True)
        plot_line(ax2, lipid, lipid_entries, extractor=extract_peak_width, min_peak = True)
        

        

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
