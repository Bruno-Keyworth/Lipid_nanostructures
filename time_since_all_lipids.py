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
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


aging_FOLDER = DATA_FOLDER / "aging"
extrusion = 31
temperatures = [10, 20, 30, 40, 50, 60]
time_format = "%d %B %Y %H:%M:%S"
LIPIDS = ["7 DMPC : 3 DMPG", "DMPC", "DMPG", "POPC", "DPPC"]
colours= {
    "DMPC": 'tab:red',
    "POPC": 'tab:blue',
    "DPPC": 'tab:purple',
    "7 DMPC : 3 DMPG": 'k'
    }
labels = {
    "DMPC": 'DMPC',
    "POPC": 'POPC',
    "DPPC": 'DPPC',
    "7 DMPC : 3 DMPG": 'DMPC:DMPG (7:3)'
    }
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

        if entry.get("type") == "size":

            if not entry.get("peaks"):
                continue

        elif entry.get("type") == "zeta":

            if entry.get("zeta_mV") is None:
                continue

        try:
            day = time_since(start, ts)
            dt = datetime.strptime(ts, time_format)

        except Exception:
            continue

        base, rep = parse_sample_name(sample)

        key = (base, rep, day, entry.get("type"))

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

def plot_line(ax, name, entries, extractor, fit):

    values_by_day = defaultdict(list)

    # Group values by day
    for entry in entries:

        day = time_since(entry['lipid_start_time'], entry['timestamp'])

        value = extractor(entry)

        if value is None or np.isnan(value):
            continue

        values_by_day[day].append(value)

    # averages
    days_sorted = np.array(sorted(values_by_day.keys()))

    avg_values = np.array([
        np.mean(values_by_day[d])
        for d in days_sorted
    ])

    std_values = np.array([
        np.std(values_by_day[d])
        for d in days_sorted
    ])
    
    avg_values -= avg_values[0]

    # plot data
    ax.errorbar(
        days_sorted,
        avg_values,
        yerr=std_values,
        fmt='o',
        ls='none',
        capsize=4,
        label=labels[name],
        c=colours[name]
    )
    if name == "7 DMPC : 3 DMPG":
        ax.plot(
            days_sorted,
            avg_values,
            ls='-',
            c=colours[name],
            lw=1
        )

    # linear fit
    if not fit:
        
        return None
    if len(days_sorted) >= 2:
        
        p, cov = np.polyfit(days_sorted, avg_values, 1, cov=True)
        x_fit = np.linspace(
            days_sorted.min(),
            days_sorted.max(),
            200
        )

        y_fit = np.polyval(p, x_fit)

        ax.plot(x_fit, y_fit, alpha=0.8, ls='--', c=colours[name])

        print(f"\n{name}")
        print(p[0])
        print(np.sqrt(cov[0][0]))

def extract_zeta(entry):

    if entry.get("type") != "zeta":
        return None

    try:
        return float(entry.get("zeta_mV", np.nan))
    except Exception:
        return None


def plot_zeta(ax, entries):

    values_by_day = defaultdict(list)

    for entry in entries:

        # only zeta entries
        if entry.get("type") != "zeta":
            continue

        value = extract_zeta(entry)

        if value is None or np.isnan(value):
            continue

        day = time_since(
            entry['lipid_start_time'],
            entry['timestamp']
        )

        values_by_day[day].append(value)

    if len(values_by_day) == 0:
        return None

    days_sorted = np.array(sorted(values_by_day.keys()))

    avg_values = np.array([
        np.mean(values_by_day[d])
        for d in days_sorted
    ])

    std_values = np.array([
        np.std(values_by_day[d], ddof=1)
        if len(values_by_day[d]) > 1 else 0
        for d in days_sorted
    ])

    ax.errorbar(
        days_sorted,
        avg_values,
        yerr=std_values,
        fmt='o',
        ls='-',
        capsize=4,
        c='k',
        label='DMPC:DMPG (7:3)'
    )


def plot_aging_lipids(entries):

    fig = plt.figure(figsize=(12, 9))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    entries_by_lipid = defaultdict(list)

    # ----------------------------
    # group by lipid
    # ----------------------------

    for entry in entries:

        lipid = entry.get("lipid")

        if lipid:
            entries_by_lipid[lipid].append(entry)

    # ----------------------------
    # assign start dates
    # ----------------------------

    for lipid, lipid_entries in entries_by_lipid.items():

        lipid_start_time = find_lipid_start(lipid_entries)

        for entry in lipid_entries:
            entry["lipid_start_time"] = lipid_start_time

    # ----------------------------
    # deduplicate
    # ----------------------------

    for lipid in entries_by_lipid:

        entries_by_lipid[lipid] = deduplicate_latest_per_day(
            entries_by_lipid[lipid]
        )

    # ----------------------------
    # diameter + width plots
    # ----------------------------

    for lipid, lipid_entries in entries_by_lipid.items():

        fit_allowed = lipid != "7 DMPC : 3 DMPG"

        plot_line(
            ax1,
            lipid,
            lipid_entries,
            extractor=extract_peak_diameter,
            fit=fit_allowed
        )

        plot_line(
            ax2,
            lipid,
            lipid_entries,
            extractor=extract_peak_width,
            fit=fit_allowed
        )

    # ----------------------------
    # zeta plot
    # ----------------------------

    if "7 DMPC : 3 DMPG" in entries_by_lipid:

        plot_zeta(
            ax3,
            entries_by_lipid["7 DMPC : 3 DMPG"]
        )

    # ----------------------------
    # labels
    # ----------------------------

    for ax in [ax1, ax2, ax3]:

        ax.set_xlabel("Time (days)", fontsize=22)
        ax.tick_params(labelsize=16)

    ax1.set_ylabel(r"$\Delta D$ (nm)", fontsize=22)
    ax2.set_ylabel(r"$\Delta W$ (nm)", fontsize=22)
    ax3.set_ylabel(r"$\zeta$ (mV)", fontsize=22)

    handles, labels = ax1.get_legend_handles_labels()

    for label, ax in {r'\textbf{(a)}': ax1, 
                      r'\textbf{(b)}': ax2, 
                      r'\textbf{(c)}' : ax3}.items(): 
        ax.text(
                0.02, 0.97, label,
                transform=ax.transAxes,
                fontsize=20,
                fontweight='bold',
                va='top', ha='left'
            )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0)
    )
    ax3.set_box_aspect(0.35)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(
        PLOTS_FOLDER / "time_since_extrusion_all_lipids.png",
        dpi=300
    )

    plt.show()

    return None

plot_aging_lipids( all_entries)
