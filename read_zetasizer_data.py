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
    Reads a Zetasizer CSV (size, zeta, or combined) and saves/merges JSON per sample.

    Preserves original functionality:
        - Uses DATA_FOLDER
        - Uses base_sample_name
        - Uses _safe_filename
        - Skips duplicates
        - Parses size peaks, zeta, etc.
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

    # Skip rows without a valid Type
    df = df[df["Type"].notna()]

    # Normalise type to lowercase
    df["Type"] = df["Type"].str.strip().str.lower()

    # Ensure output directory exists
    out_dir = DATA_FOLDER / save_to_folder
    os.makedirs(out_dir, exist_ok=True)

    # Group by base sample name
    df["Base Sample Name"] = df["Sample Name"].apply(base_sample_name)
    grouped = df.groupby("Base Sample Name")

    for base_name, group in grouped:

        file_path = out_dir / (_safe_filename(base_name) + ".json")
        data = []
        existing_timestamps = set()

        # Load existing data if present
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
                "timestamp": timestamp,
                "temperature_C": row.get("T"),
            }

            if row["Type"] == "size":
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

            elif row["Type"] == "zeta":
                entry.update({
                    "type": "zeta",
                    "zeta_mV": float(row.get("ZP")),
                    "mobility_umcm_Vs": float(row.get("Mob")),
                    "conductivity_mScm": float(row.get("Cond"))
                })

            else:
                print(f"Cannot read measurement type: {row['Type']}")
                continue

            existing_timestamps.add(timestamp)
            data.append(entry)

        # Write merged JSON back
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
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