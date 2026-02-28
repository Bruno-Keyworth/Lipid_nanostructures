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
from get_filepaths import _get_file, DATA_FOLDER

def process_dls_csv(
    csv_path,
    get_file=_get_file,
    extrusions=(3, 5, 10, 15, 20, 31, 41),
    temperatures=(10, 20, 30, 40, 50, 60),
    encoding="latin1",
    sep="\t"
):
    """
    Read a DLS CSV file, group data by extrusion and temperature,
    and write processed JSON files using get_file(t, e).

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file.
    get_file : callable
        Function get_file(temperature, extrusion) -> output filepath.
    extrusions : iterable
        Extrusion values to process.
    temperatures : iterable
        Temperature values to process.
    encoding : str
        File encoding.
    sep : str
        Column separator.
    """

    # ----------------------------
    # Read CSV
    # ----------------------------
    df = pd.read_csv(
        DATA_FOLDER / csv_path,
        encoding=encoding,
        sep=sep,
        engine="python",
        on_bad_lines="warn"
    )

    # ----------------------------
    # Identify column groups
    # ----------------------------
    size_cols = [c for c in df.columns if c.startswith("Sizes[")]
    intensity_cols = [c for c in df.columns if c.startswith("Intensities[")]
    corr_cols = [c for c in df.columns if c.startswith("Correlation Data[")]
    delay_cols = [c for c in df.columns if c.startswith("Correlation Delay Times[")]
    volume_cols = [c for c in df.columns if c.startswith("Volumes[")]

    scalar_cols = [
        c for c in df.columns
        if c not in size_cols + intensity_cols + corr_cols + delay_cols + volume_cols
    ]

    meta = df[scalar_cols].copy()

    # ----------------------------
    # Clean rows
    # ----------------------------
    df_valid = df[df["Sample Name"] != "Sample Name"]
    df_valid = df_valid.drop_duplicates(
        subset=["Sample Name", "Measurement Date and Time"]
    )

    # ----------------------------
    # Loop over conditions
    # ----------------------------
    for e in extrusions:
        for t in temperatures:
            mask = df_valid["Sample Name"].str.split().apply(
                lambda s: len(s) > 5 and s[0] == str(e) and s[5] == str(t)
            )

            subset = df_valid[mask]
            file = get_file(t, e)

            print(f"Processing T={t}, extrusion={e}")

            data = []

            for i, row in subset.iterrows():
                meta_dict = meta.iloc[i].to_dict()

                size_row = row[size_cols].to_numpy()
                intensity_row = row[intensity_cols].to_numpy()
                volume_row = row[volume_cols].to_numpy()
                correlation_row = row[corr_cols].to_numpy()
                delay_row = row[delay_cols].to_numpy()

                data.append({
                    "meta": meta_dict,
                    "size": np.column_stack(
                        (size_row, intensity_row, volume_row)
                    ).T.tolist(),
                    "correlation": np.column_stack(
                        (correlation_row, delay_row)
                    ).T.tolist(),
                })

            if data:
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    
                    
                    
process_dls_csv("data[1].csv")