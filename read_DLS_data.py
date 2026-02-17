#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:35:58 2026

@author: brunokeyworth
"""

import pandas as pd
import numpy as np
from get_filepaths import get_file
import os
import json

df = pd.read_csv(
    "../Data/data[1].csv",
    encoding="latin1",
    sep="\t",
    engine="python"
)

df.shape
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


extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]
df_valid = df[df["Sample Name"] != "Sample Name"]
df_valid = df_valid.drop_duplicates(subset=["Sample Name", "Measurement Date and Time"])
for e in extrusions:
    for t in temperatures:
        mask = df_valid["Sample Name"].str.split().apply(
            lambda s: s[0] == str(e) and s[5] == str(t)
        )
        subset = df_valid[mask]
        file = get_file(t, e)
        print(t, e)
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
                "size": np.column_stack((size_row, intensity_row, volume_row)).T.tolist(),
                "correlation": np.column_stack((correlation_row, delay_row)).T.tolist(),
            })
        if data:  # only save if there is data
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)