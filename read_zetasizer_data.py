#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:35:58 2026

@author: brunokeyworth
"""
import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from get_filepaths import DATA_FOLDER

def _parse_array(cell):
    """Convert space-separated numeric string to numpy array."""
    if pd.isna(cell):
        return np.array([])
    values = []
    for x in str(cell).split():
        try:
            values.append(float(x))
        except ValueError:
            continue
    return np.array(values)

def _safe_filename(name):
    """Make a filesystem-safe filename."""
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "").replace(':', '')
    )

def base_sample_name(name):
    # ensure space before mg_ml
    name = re.sub(r'(?<!\s)(mg_ml)', r' \1', name)
    
    # ensure space after surfactant names
    name = re.sub(r'(C12E6|TX100|DDAC)(?!\s)', r'\1 ', name)
    
    # normalise multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    parts = str(name).split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)

def read_zetasizer_data(csv_path, save_to_folder, encoding="latin1", sep="\t"):

    df = pd.read_csv(
        DATA_FOLDER / csv_path,
        encoding=encoding,
        sep=sep,
        engine="python",
        skiprows=[1],  # skip units row
        on_bad_lines="warn"
    )

    df = df[df["Sample Name"].notna()]
    df = df[df["Sample Name"] != "Sample Name"]

    out_dir = DATA_FOLDER / save_to_folder
    os.makedirs(out_dir, exist_ok=True)

    df["Base Sample Name"] = df["Sample Name"].apply(base_sample_name)
    grouped = df.groupby("Base Sample Name")

    for base_name, group in grouped:

        # Load existing data if file exists (so size + zeta can merge)
        file_path = out_dir / (_safe_filename(base_name) + ".json")
        data = []
        data = []
        existing_timestamps = set()
        
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_timestamps = {
                    entry.get("timestamp") for entry in data if entry.get("timestamp") is not None
                }

        for _, row in group.iterrows():
            timestamp = row.get("Measurement Date and Time")

            # Skip duplicates
            if timestamp in existing_timestamps:
                continue
            entry = {
                "sample_name": row["Sample Name"],
                "timestamp": row.get("Measurement Date and Time"),
                "temperature_C": row.get("T"),
            }

            if row["Type"].strip().lower() == "size":
                sizes = _parse_array(row.get("Sizes"))
                intensities = _parse_array(row.get("Intensities"))
                entry.update({
                    "type": "size",
                    "peaks": [
                        {
                            "peak_position_nm": row.get("Pk 1 Mean Int"),
                            "area_percent": row.get("Pk 1 Area Int"),
                            "peak_width_nm": row.get("Size Peak")
                        },
                        {
                            "peak_position_nm": row.get("Pk 2 Mean Int"),
                            "area_percent": row.get("Pk 2 Area Int"),
                            "peak_width_nm": row.get("Size Peak.1")
                        },
                        {
                            "peak_position_nm": row.get("Pk 3 Mean Int"),
                            "area_percent": row.get("Pk 3 Area Int"),
                            "peak_width_nm": row.get("Size Peak.2")
                        },
                    ],
                    "sizes_nm": sizes.tolist(),
                    "intensities_percent": intensities.tolist(),
                    "z_average_nm": row.get("Z-Ave"),
                    "pdi": row.get("PdI")
                })
            elif row["Type"].strip().lower() == "zeta":
                entry.update({
                    "type": "zeta",
                    "zeta_mV": row.get("ZP"),
                    "mobility_umcm_Vs": row.get("Mob"),
                    "conductivity_mScm": row.get("Cond")
                })
            else:
                print(f"Cannot read measurement type: {row['Type'].strip().lower()}")
                continue

            # store remaining metadata
            # entry["metadata"] = {
            #     k: row[k]
            #     for k in row.index
            #     if k not in [
            #         "Sample Name",
            #         "Base Sample Name",
            #         "Measurement Date and Time",
            #         "T",
            #         "Sizes",
            #         "Intensities",
            #         "Type"
            #     ]
            # }
            existing_timestamps.add(timestamp)
            data.append(entry)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)