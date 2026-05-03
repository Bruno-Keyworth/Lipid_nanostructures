#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:13:22 2026

@author: brunokeyworth
"""

import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from get_filepaths import DATA_FOLDER


# =========================
# INTERACTIVE NAME HANDLING
# =========================

def resolve_sample_name(base_name, name_map, interactive=True):
    """
    Prompts user to confirm or rename a base sample name.
    Uses a cache so each name is only handled once.
    """
    if base_name in name_map:
        return name_map[base_name]

    if not interactive:
        name_map[base_name] = base_name
        return base_name

    print(f"\nDetected sample: '{base_name}'")
    new_name = input("Rename? (Enter = keep, or type new name): ").strip()

    final_name = new_name if new_name else base_name
    name_map[base_name] = final_name

    return final_name


def load_name_map(mapping_file):
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_name_map(mapping_file, name_map):
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2)


# =========================
# FILE SPLITTING
# =========================

def split_zeta_size_file(input_path, output_dir):
    output_dir = DATA_FOLDER / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    size_lines = []
    zeta_lines = []

    current_block = None
    keep_block = False

    with open(input_path, "r", encoding="latin1") as f:
        for line in f:
            line_strip = line.strip()

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

            if line_strip.startswith(("Mean", "Minimum", "Maximum")):
                keep_block = False
                continue

            if not keep_block:
                continue

            if current_block == "size":
                size_lines.append(line)
            elif current_block == "zeta":
                zeta_lines.append(line)

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


# =========================
# HELPERS
# =========================

def _parse_array(cell):
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
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
    )


def base_sample_name(name):
    name = re.sub(r'(?<!\s)(mg_ml)', r' \1', name)
    name = re.sub(r'(C12E6|TX100|DDAC)(?!\s)', r'\1 ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    parts = str(name).split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]

    return " ".join(parts)


# =========================
# MAIN PARSER
# =========================

def read_zetasizer_file(csv_path, save_to_folder, name_map, interactive=True,
                       encoding="latin1", sep="\t"):

    df = pd.read_csv(
        csv_path,
        encoding=encoding,
        sep=sep,
        engine="python",
        skiprows=[1],
        on_bad_lines="warn"
    )

    df = df[df["Sample Name"].notna()]
    df = df[df["Sample Name"] != "Sample Name"]
    df = df[df["Type"].notna()]

    df["Type"] = df["Type"].str.strip().str.lower()
    df["Record"] = pd.to_numeric(df["Record"], errors='coerce')

    # === INTERACTIVE RENAMING HERE ===
    df["Base Sample Name"] = df["Sample Name"].apply(
        lambda x: resolve_sample_name(
            base_sample_name(x),
            name_map,
            interactive=interactive
        )
    )

    grouped = df.groupby("Base Sample Name")

    out_dir = DATA_FOLDER / save_to_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for base_name, group in grouped:
        file_path = out_dir / (_safe_filename(base_name) + ".json")

        timestamp_map = {}

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

            if timestamp in timestamp_map:
                if record_num <= timestamp_map[timestamp][0]:
                    continue

            if save_to_folder.lower() != "aging":
                if "aging" in sample_name.lower():
                    continue

            entry = {
                "sample_name": sample_name,
                "record_number": record_num,
                "timestamp": timestamp,
                "temperature_C": row.get("T"),
            }

            if row["Type"] == "size":
                entry.update({
                    "type": "size",
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

        if timestamp_map:
            final_data = [val[1] for val in timestamp_map.values()]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2)


# =========================
# PIPELINE ENTRY POINT
# =========================

def read_data_from_kate(file, output_folder, interactive=True):
    file = DATA_FOLDER / file

    mapping_file = DATA_FOLDER / "sample_name_map.json"
    name_map = load_name_map(mapping_file)

    print(f"Processing: {file.name}")

    size_file, zeta_file = split_zeta_size_file(file, output_folder)

    if size_file.exists():
        read_zetasizer_file(
            csv_path=size_file,
            save_to_folder=output_folder,
            name_map=name_map,
            interactive=interactive
        )

    if zeta_file.exists():
        read_zetasizer_file(
            csv_path=zeta_file,
            save_to_folder=output_folder,
            name_map=name_map,
            interactive=interactive
        )

    save_name_map(mapping_file, name_map)
