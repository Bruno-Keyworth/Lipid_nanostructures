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
import pickle

df = pd.read_csv(
    "../Data/data.csv",
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

sizes = df[size_cols].to_numpy()
intensities = df[intensity_cols].to_numpy()
volumes = df[volume_cols].to_numpy()
correlations = df[corr_cols].to_numpy()
delays = df[delay_cols].to_numpy()

extrusions = [3, 5, 10, 15, 20, 31, 41]
temperatures = [10, 20, 30, 40, 50, 60]

for e in extrusions:
    for t in temperatures:
        file = get_file(t, e)
        data = []
        for i, row in df.iterrows():
            if row["Sample Name"] == "Sample Name":
                continue
            sample_name = row["Sample Name"].split(' ')
            if not ((sample_name[0] == str(e)) and (sample_name[5] == str(t))):
                continue
            meta_dict = meta.iloc[i].to_dict()
            data.append({'datetime': row["Measurement Date and Time"],
                    'peak': [],
                    'size': np.column_stack((sizes[i], intensities[i], volumes[i])),
                    'correlation': np.column_stack((correlations[i], delays[i])),
                    'meta': meta_dict,
                    })
        if data:  # only save if there is data
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'wb') as f:
                pickle.dump(data, f)