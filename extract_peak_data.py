#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:39:30 2026

@author: brunokeyworth
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from get_filepaths import DATA_FOLDER


def parse_timestamp(ts):
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%d/%m/%Y %H:%M:%S")
        except Exception:
            return None


def base_sample_name(name):
    parts = str(name).split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def average_peaks_from_file(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [d for d in data if d.get("type") == "size"]

    # keep latest measurement for identical sample + temperature
    latest = {}

    for entry in data:
        key = (entry["sample_name"], entry["temperature_C"])
        ts = parse_timestamp(entry["timestamp"])

        if key not in latest:
            latest[key] = entry
        else:
            old_ts = parse_timestamp(latest[key]["timestamp"])
            if ts and old_ts and ts > old_ts:
                latest[key] = entry

    filtered = list(latest.values())

    groups = defaultdict(list)

    for entry in filtered:
        base = base_sample_name(entry["sample_name"])
        key = (base, entry["temperature_C"])
        groups[key].append(entry)

    results = []

    for (base, temp), entries in groups.items():

        ranked_peaks = [[], [], []]

        for entry in entries:

            peaks = [p for p in entry["peaks"] if p["peak_position_nm"] is not None]

            peaks_sorted = sorted(
                peaks,
                key=lambda p: p["area_percent"] if p["area_percent"] else 0,
                reverse=True
            )

            for i in range(min(3, len(peaks_sorted))):
                ranked_peaks[i].append(peaks_sorted[i])

        avg_peaks = []

        for peak_list in ranked_peaks:
            if not peak_list:
                continue

            avg_peaks.append({
                "mean_peak_position_nm": float(np.mean([p["peak_position_nm"] for p in peak_list])),
                "mean_area_percent": float(np.mean([p["area_percent"] for p in peak_list])),
                "mean_peak_width_nm": float(np.mean([p["peak_width_nm"] for p in peak_list])),
                "n_measurements": len(peak_list)
            })

        results.append({
            "base_sample_name": base,
            "temperature_C": temp,
            "averaged_peaks": avg_peaks
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def process_folder(folder):

    folder = DATA_FOLDER / folder
    out_folder = folder / "averaged_peaks"
    out_folder.mkdir(exist_ok=True)

    for file in folder.glob("*.json"):

        output_file = out_folder / file.name
        average_peaks_from_file(file, output_file)

        print(f"Processed {file.name}")


# example
process_folder("POPC-POPG")
process_folder("surfactants")