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
        .replace(")", "")
    )


def process_dls_csv(csv_path, encoding="latin1", sep="\t"):

    df = pd.read_csv(
    DATA_FOLDER / csv_path,
    encoding=encoding,
    sep=sep,
    engine="python",
    skiprows=[1],   # skip units row
    on_bad_lines="warn"
)

    # Remove header/unit duplicates if present
    df = df[df["Sample Name"].notna()]
    df = df[df["Sample Name"] != "Sample Name"]

    # Output directory
    out_dir = DATA_FOLDER / "POPC"
    os.makedirs(out_dir, exist_ok=True)

    grouped = df.groupby("Sample Name")

    for sample_name, group in grouped:

        data = []

        for _, row in group.iterrows():

            sizes = _parse_array(row["Sizes"])
            intensities = _parse_array(row["Intensities"])

            entry = {
                "sample_name": sample_name,
                "timestamp": row.get("Measurement Date and Time"),
                "temperature_C": row.get("T"),

                "sizes_nm": sizes.tolist(),
                "intensities_percent": intensities.tolist(),

                "peaks": [
                    {
                        "mean_nm": row.get("Pk 1 Mean Int"),
                        "area_percent": row.get("Pk 1 Area Int"),
                        "size_peak_nm": row.get("Size Peak")
                    },
                    {
                        "mean_nm": row.get("Pk 2 Mean Int"),
                        "area_percent": row.get("Pk 2 Area Int"),
                        "size_peak_nm": row.get("Size Peak.1")
                    },
                    {
                        "mean_nm": row.get("Pk 3 Mean Int"),
                        "area_percent": row.get("Pk 3 Area Int"),
                        "size_peak_nm": row.get("Size Peak.2")
                    },
                ],

                "z_average_nm": row.get("Z-Ave"),
                "pdi": row.get("PdI"),

                # store remaining metadata as well
                "metadata": {
                    k: row[k]
                    for k in row.index
                    if k not in [
                        "Sample Name",
                        "Measurement Date and Time",
                        "T",
                        "Sizes",
                        "Intensities"
                    ]
                }
            }

            data.append(entry)

        file_name = _safe_filename(sample_name) + ".json"
        file_path = out_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


process_dls_csv("POPC_temp_extrusion_size.txt")