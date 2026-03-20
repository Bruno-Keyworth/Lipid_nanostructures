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
import pandas as pd

from sklearn.cluster import DBSCAN

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

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def _output_file(input_file, temp):

    out_folder = input_file.parent / f"{int(float(temp))}_degrees"
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

def extract_valid_peaks(entry, min_area, max_pos_nm):
    peaks = []

    for p in entry.get("peaks", []):
        pos = safe_float(p.get("peak_position_nm"))
        width = safe_float(p.get("peak_width_nm"))
        area = safe_float(p.get("area_percent"))

        if (
            pos is None or width is None or area is None
            or pos <= 0 or width <= 0
            or area <= min_area
            or pos > max_pos_nm
        ):
            continue

        peaks.append({
            "peak_position_nm": pos,
            "peak_width_nm": width,
            "area_percent": area,
        })

    return peaks

def renormalise_peaks(peaks):
    if not peaks:
        return []

    areas = np.array([p["area_percent"] for p in peaks])
    total = areas.sum()

    if total == 0:
        return []

    scale = 100 / total

    for p in peaks:
        p["area_percent"] *= scale

    return peaks

def collect_all_peaks(entries, min_area, max_pos_nm):
    all_peaks = []
    peak_owner = []

    for idx, entry in enumerate(entries):
        peaks = extract_valid_peaks(entry, min_area, max_pos_nm)
        peaks = renormalise_peaks(peaks)

        for p in peaks:
            all_peaks.append(p)
            peak_owner.append(idx)

    return all_peaks, peak_owner

def build_distance_matrix(positions, widths):
    n = len(positions)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            width_scale = 0.25 * (widths[i] + widths[j])
            dist[i, j] = abs(positions[i] - positions[j]) / width_scale

    return dist

def compute_cluster_stats(clusters, peak_owner, n_entries):
    results = []

    for cluster in clusters.values():
        pos_vals = [p["peak_position_nm"] for p, _ in cluster]
        wid_vals = [p["peak_width_nm"] for p, _ in cluster]

        pos_mean = float(np.mean(pos_vals))
        pos_std = float(np.std(pos_vals, ddof=1)) if len(pos_vals) > 1 else 0.0

        wid_mean = float(np.mean(wid_vals))
        wid_std = float(np.std(wid_vals, ddof=1)) if len(wid_vals) > 1 else 0.0

        per_entry_area = np.zeros(n_entries)

        for p, owner in cluster:
            per_entry_area[owner] += p["area_percent"]

        area_mean = float(np.mean(per_entry_area))
        area_std = float(np.std(per_entry_area, ddof=1)) if np.count_nonzero(per_entry_area) > 1 else 0.0

        results.append({
            "peak_position_nm": [pos_mean, pos_std],
            "peak_width_nm": [wid_mean, wid_std],
            "area_percent": [area_mean, area_std],
            "n_measurements": int(np.count_nonzero(per_entry_area)),
        })

    return results

def cluster_peaks(entries, max_pos_nm=3000, min_area=7):

    all_peaks, peak_owner = collect_all_peaks(entries, min_area, max_pos_nm)

    if not all_peaks:
        return []

    positions = np.array([p["peak_position_nm"] for p in all_peaks])
    widths = np.array([p["peak_width_nm"] for p in all_peaks])

    dist = build_distance_matrix(positions, widths)

    labels = DBSCAN(eps=1.0, min_samples=1, metric="precomputed").fit(dist).labels_

    # group clusters
    clusters = {}
    for label, peak, owner in zip(labels, all_peaks, peak_owner):
        clusters.setdefault(label, []).append((peak, owner))

    results = compute_cluster_stats(clusters, peak_owner, len(entries))

    # final renormalisation
    total_area = sum(r["area_percent"][0] for r in results)
    if total_area > 0:
        for r in results:
            r["area_percent"][0] *= 100 / total_area
            r["area_percent"][1] *= 100 / total_area

    return results

def average_measurements(input_file):

    import json
    import numpy as np

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- Split by type ----
    size_data = [d for d in data if d.get("type") == "size"]
    zeta_data = [d for d in data if d.get("type") == "zeta"]

    # ---- Keep only latest measurements ----
    size_data = filter_latest_measurements(size_data)
    zeta_data = filter_latest_measurements(zeta_data)

    # ---- Group ----
    size_groups = group_by_base_and_temp(size_data)
    zeta_groups = group_by_base_and_temp(zeta_data)

    all_keys = set(size_groups) | set(zeta_groups)

    # ---- Process each condition ----
    for (base, temp) in all_keys:

        # ---------- SIZE / PEAKS ----------
        entries = size_groups.get((base, temp), [])

        repeat_peaks = [
            [
                p for p in entry["peaks"]
                if p["peak_position_nm"] is not None
                and float(p.get("area_percent") or 0) > 0
            ]
            for entry in entries
        ]

        # This now returns FINAL peaks with [mean, std]
        avg_peaks = cluster_peaks(entries)

        # Sanity check (optional but useful)
        if avg_peaks:
            total_area = sum(p.get("area_percent")[0] for p in avg_peaks)
            if not np.isclose(total_area, 100, atol=1e-6):
                print(f"Warning: area sums to {total_area:.2f} for {base}, {temp}")

        # ---------- ZETA ----------
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

        # ---------- OUTPUT ----------
        results = read_sample_name(base) | {
            "temperature_C": temp,
            "average_zeta": average_zeta,
            "averaged_peaks": avg_peaks,
            "repeat_zetas": repeat_zetas,
            "repeat_peaks": repeat_peaks,
        }

        with open(_output_file(input_file, temp), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

def mean_err(entry):
    if entry and isinstance(entry, list) and len(entry) == 2:
        return entry[0], entry[1]
    return np.nan, np.nan

def gather_data(folder):

    records = []

    for fp in folder.glob("*.json"):
        with open(fp) as f:
            data = json.load(f)

        if not np.isclose(data.get("lipid_conc_mg_ml", np.nan), 0.1):
            continue

        lipid = data["lipid_ratio"]
        surf = data["surfactant_conc_microM"]

        DMPC = lipid.get("DMPC", 0)
        DMPG = lipid.get("DMPG", 0)

        if DMPC == 0 and DMPG == 0:
            continue

        total = DMPC + DMPG
        frac_DMPG = DMPG / total if total > 0 else np.nan

        surf_total = surf.get("C12E6", 0) + surf.get("DDAC", 0) + surf.get("TX100", 0)

        peaks = data.get("averaged_peaks", [])

        p1 = peaks[0] if len(peaks) > 0 else {}
        p2 = peaks[1] if len(peaks) > 1 else {}
        p3 = peaks[2] if len(peaks) > 2 else {}

        zeta, zeta_err = mean_err(data.get("average_zeta"))

        p1_pos, p1_pos_err = mean_err(p1.get("peak_position_nm"))
        p1_width, p1_width_err = mean_err(p1.get("peak_width_nm"))
        p1_area, p1_area_err = mean_err(p1.get("area_percent"))

        p2_pos, p2_pos_err = mean_err(p2.get("peak_position_nm"))
        p2_width, p2_width_err = mean_err(p2.get("peak_width_nm"))
        p2_area, p2_area_err = mean_err(p2.get("area_percent"))

        p3_pos, p3_pos_err = mean_err(p3.get("peak_position_nm"))
        p3_width, p3_width_err = mean_err(p3.get("peak_width_nm"))
        p3_area, p3_area_err = mean_err(p3.get("area_percent"))

        record = {
            "fraction_DMPG": frac_DMPG,
            "surfactant_total": surf_total,

            "zeta": zeta,
            "zeta_err": zeta_err,

            "p1_pos": p1_pos,
            "p1_pos_err": p1_pos_err,
            "p1_width": p1_width,
            "p1_width_err": p1_width_err,
            "p1_area": p1_area,
            "p1_area_err": p1_area_err,

            "p2_pos": p2_pos,
            "p2_pos_err": p2_pos_err,
            "p2_width": p2_width,
            "p2_width_err": p2_width_err,
            "p2_area": p2_area,
            "p2_area_err": p2_area_err,

            "p3_pos": p3_pos,
            "p3_pos_err": p3_pos_err,
            "p3_width": p3_width,
            "p3_width_err": p3_width_err,
            "p3_area": p3_area,
            "p3_area_err": p3_area_err,

            "C12E6": surf.get("C12E6", 0),
            "DDAC": surf.get("DDAC", 0),
            "TX100": surf.get("TX100", 0),
        }

        records.append(record)

    return pd.DataFrame(records)

def process_folder(folder):

    folder = DATA_FOLDER / folder

    for file in folder.glob("*.json"):
        average_measurements(file)


if __name__ == '__main__':

    read_zetasizer_data("POPC_POPG_fraction_sizes.txt", "POPC-POPG")
    read_zetasizer_data("POPC_POPG_zetas.txt", "POPC-POPG")
    read_zetasizer_data("POPC_temp_extrsusion_size.txt", "POPC")
    read_zetasizer_data("surfactant_sizes.txt", "surfactants")
    read_zetasizer_data("surfactant_zetas.txt", "surfactants")
    read_zetasizer_data('data_from_kate.txt', 'data_from_kate')
    read_zetasizer_data('time_since_extrusion.txt', 'POPC')

    process_folder("POPC")
    process_folder("POPC-POPG")
    process_folder("surfactants")
    process_folder("data_from_kate")