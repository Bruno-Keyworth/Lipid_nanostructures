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

def split_zeta_size_file(input_path, output_dir):
    """
    Splits a raw Zetasizer export into two files:
        - *_size.txt
        - *_zeta.txt

    Handles multiple blocks and ignores summary sections.
    """
    output_dir = DATA_FOLDER / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    size_lines = []
    zeta_lines = []

    current_block = None
    keep_block = False

    with open(input_path, "r", encoding="latin1") as f:
        for line in f:
            line_strip = line.strip()

            # Detect new header
            if line_strip.startswith("Record\tType"):
                if "Z-Ave" in line:
                    current_block = "size"
                elif "ZP" in line:
                    current_block = "zeta"
                else:
                    current_block = None

                keep_block = True

                if current_block == "size":
                    size_lines.append(line)
                elif current_block == "zeta":
                    zeta_lines.append(line)

                continue

            # Skip summary sections
            if line_strip.startswith(("Mean", "Minimum", "Maximum")):
                keep_block = False
                continue

            if not keep_block:
                continue

            if current_block == "size":
                size_lines.append(line)
            elif current_block == "zeta":
                zeta_lines.append(line)

    # Write outputs
    base = Path(input_path).stem

    size_file = output_dir / f"{base}_size.txt"
    zeta_file = output_dir / f"{base}_zeta.txt"

    if size_lines:
        with open(size_file, "w", encoding="utf-8") as f:
            f.writelines(size_lines)

    if zeta_lines:
        with open(zeta_file, "w", encoding="utf-8") as f:
            f.writelines(zeta_lines)

    return size_file, zeta_file

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

def read_zetasizer_file(csv_path, save_to_folder, encoding="latin1", sep="\t"):
    """
    Reads a Zetasizer CSV and saves/merges JSON per sample.
    
    Updated Logic: 
    - Always saves to the root of save_to_folder (no 'aging' subfolder).
    - If save_to_folder != 'aging', skips any sample entries containing 'aging'.
    - Filters by highest record per timestamp.
    """

    # Load CSV from DATA_FOLDER
    df = pd.read_csv(
        csv_path,
        encoding=encoding,
        sep=sep,
        engine="python",
        skiprows=[1],  # skip units row
        on_bad_lines="warn"
    )

    # Remove empty rows or repeated headers
    df = df[df["Sample Name"].notna()]
    df = df[df["Sample Name"] != "Sample Name"]
    df = df[df["Type"].notna()]
    df["Type"] = df["Type"].str.strip().str.lower()
    df["Record"] = pd.to_numeric(df["Record"], errors='coerce')

    # Group by base sample name
    df["Base Sample Name"] = df["Sample Name"].apply(base_sample_name)
    grouped = df.groupby("Base Sample Name")

    # Set the output directory once - no subfolders
    out_dir = DATA_FOLDER / save_to_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for base_name, group in grouped:
        # Define file path directly in the main out_dir
        file_path = out_dir / (_safe_filename(base_name) + ".json")

        # Track the highest Record per timestamp
        timestamp_map = {}

        # Load existing data to populate the map
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for entry in existing_data:
                    ts = entry.get("timestamp")
                    rec = entry.get("record_number", 0)
                    timestamp_map[ts] = (rec, entry)

        for _, row in group.iterrows():
            timestamp = row.get("Measurement Date and Time")
            record_num = int(row.get("Record", 0))
            sample_name = row["Sample Name"]

            # 1. TIE-BREAKER LOGIC: 
            # Reject if record number isn't newer/higher for this timestamp
            if timestamp in timestamp_map:
                if record_num <= timestamp_map[timestamp][0]:
                    continue 

            # 2. AGING FILTER LOGIC:
            # If the folder is NOT 'aging', skip any individual row with 'aging' in the name
            if save_to_folder.lower() != "aging":
                if "aging" in sample_name.lower():
                    continue

            # Build the entry
            entry = {
                "sample_name": sample_name,
                "record_number": record_num,
                "timestamp": timestamp,
                "temperature_C": row.get("T"),
            }

            if row["Type"] == "size":
                sizes = _parse_array(row.get("Sizes"))
                intensities = _parse_array(row.get("Intensities"))
                entry.update({
                    "type": "size",
                    "peaks": [
                        {"peak_position_nm": row.get("Pk 1 Mean Int"), "area_percent": row.get("Pk 1 Area Int"), "peak_width_nm": row.get("Size Peak")},
                        {"peak_position_nm": row.get("Pk 2 Mean Int"), "area_percent": row.get("Pk 2 Area Int"), "peak_width_nm": row.get("Size Peak.1")},
                        {"peak_position_nm": row.get("Pk 3 Mean Int"), "area_percent": row.get("Pk 3 Area Int"), "peak_width_nm": row.get("Size Peak.2")},
                    ],
                    "sizes_nm": sizes.tolist() if sizes is not None else [],
                    "intensities_percent": intensities.tolist() if intensities is not None else [],
                    "z_average_nm": row.get("Z-Ave"),
                    "pdi": row.get("PdI")
                })
            elif row["Type"] == "zeta":
                entry.update({
                    "type": "zeta",
                    "zeta_mV": float(row.get("ZP", 0)),
                    "mobility_umcm_Vs": float(row.get("Mob", 0)),
                    "conductivity_mScm": float(row.get("Cond", 0))
                })
            else:
                continue

            timestamp_map[timestamp] = (record_num, entry)

        # Convert map back to list of entries and save
        if timestamp_map:
            final_data = [val[1] for val in timestamp_map.values()]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2)
            
def read_zetasizer_data(file, output_folder):
    """
    Full pipeline:
        raw files → split into size/zeta → parsed into JSON

    Parameters
    ----------
    raw_folder : folder containing original exports
    split_folder : where split txt files go
    json_folder : where JSON output goes
    """
    file = DATA_FOLDER / file

    print(f"Processing: {file.name}")

    size_file, zeta_file = split_zeta_size_file(file, output_folder)

    # Feed BOTH into your existing parser
    if size_file.exists():
        read_zetasizer_file(
            csv_path=size_file,
            save_to_folder=output_folder
        )

    if zeta_file.exists():
        read_zetasizer_file(
            csv_path=zeta_file,
            save_to_folder=output_folder
        )