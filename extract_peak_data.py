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
from read_zetasizer_data import read_zetasizer_data, base_sample_name
from read_sample_name import read_sample_name


def parse_timestamp(ts):

    if ts is None:
        return None

    formats = [
        "%d %B %Y %H:%M:%S",   # 05 March 2026 12:28:23
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass

    return None

def _output_file(input_file, temp):

    out_folder = input_file.parent / f"{int(temp)}_degrees"
    out_folder.mkdir(exist_ok=True)

    return out_folder / input_file.name

def filter_latest_measurements(data):
    """
    Keep only the most recent measurement for each
    (sample_name, temperature_C) pair.
    """

    latest = {}

    for entry in data:

        key = (entry["sample_name"], entry["temperature_C"])
        ts = parse_timestamp(entry["timestamp"])

        if key not in latest:
            latest[key] = entry
            continue

        old_entry = latest[key]
        old_ts = parse_timestamp(old_entry["timestamp"])

        if ts is not None and old_ts is not None and ts > old_ts:
            latest[key] = entry

    return list(latest.values())

def group_by_base_and_temp(data):

    groups = defaultdict(list)

    for entry in data:

        base = base_sample_name(entry["sample_name"])
        key = (base, entry["temperature_C"])

        groups[key].append(entry)

    return groups

def cluster_peaks(entries, tol_nm=20):
    """
    Cluster peaks from repeated measurements based on position proximity.
    """

    clusters = []

    for entry in entries:

        peaks = [
            p for p in entry["peaks"]
            if p["peak_position_nm"] is not None and (p["area_percent"] or 0) > 0
        ]

        for peak in peaks:

            pos = peak["peak_position_nm"]

            placed = False

            for cluster in clusters:

                mean_pos = np.mean([p["peak_position_nm"] for p in cluster])

                if abs(pos - mean_pos) <= tol_nm:
                    cluster.append(peak)
                    placed = True
                    break

            if not placed:
                clusters.append([peak])

    return clusters


def average_measurements(input_file):

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    size_data = [d for d in data if d.get("type") == "size"]
    zeta_data = [d for d in data if d.get("type") == "zeta"]

    size_data = filter_latest_measurements(size_data)
    zeta_data = filter_latest_measurements(zeta_data)

    size_groups = group_by_base_and_temp(size_data)
    zeta_groups = group_by_base_and_temp(zeta_data)

    all_keys = set(size_groups) | set(zeta_groups)

    for (base, temp) in all_keys:

        entries = size_groups.get((base, temp), [])

        repeat_peaks = [
        [
            p for p in entry["peaks"]
            if p["peak_position_nm"] is not None and (p["area_percent"] or 0) > 0
        ]
        for entry in entries
        ]
    
        clusters = cluster_peaks(entries)

        avg_peaks = []

        for cluster in clusters:
        
            positions = [p["peak_position_nm"] for p in cluster]
            widths = [p["peak_width_nm"] for p in cluster]
            areas = [p["area_percent"] for p in cluster]
        
            avg_peaks.append({
                "peak_position_nm": [
                    float(np.mean(positions)),
                    float(np.std(positions, ddof=1)) if len(positions) > 1 else 0.0
                ],
                "peak_width_nm": [
                    float(np.mean(widths)),
                    float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0
                ],
                "area_percent": [
                    float(np.mean(areas)),
                    float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0
                ],
                "n_measurements": len(cluster)
            })


        zeta_entries = zeta_groups.get((base, temp), [])

        repeat_zetas = [
            entry["zeta_mV"]
            for entry in zeta_entries
            if entry.get("zeta_mV") is not None
        ]

        if repeat_zetas:
            average_zeta = [
                float(np.mean(repeat_zetas)),
                float(np.std(repeat_zetas, ddof=1)) if len(repeat_zetas) > 1 else 0.0
            ]
        else:
            average_zeta = [None, None]

        results = read_sample_name(base) | {
            "temperature_C": temp,
            "average_zeta": average_zeta,
            "averaged_peaks": avg_peaks,
            "repeat_zetas": repeat_zetas,
            "repeat_peaks": repeat_peaks,
        }

        with open(_output_file(input_file, temp), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def process_folder(folder):

    folder = DATA_FOLDER / folder

    for file in folder.glob("*.json"):
        average_measurements(file)


if __name__ == '__main__':

    read_zetasizer_data("POPC_POPG_fraction_sizes.txt", "POPC-POPG")
    read_zetasizer_data("POPC_POPG_zetas.txt", "POPC-POPG")
    read_zetasizer_data("POPC_temp_extrusion_size.txt", "POPC")
    read_zetasizer_data("surfactant_sizes.txt", "surfactants")
    read_zetasizer_data("surfactant_zetas.txt", "surfactants")

    process_folder("POPC")
    process_folder("POPC-POPG")
    process_folder("surfactants")