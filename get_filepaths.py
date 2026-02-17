#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:09:39 2026

@author: brunokeyworth
"""

from pathlib import Path
import socket

if socket.gethostname() == "Brunos-MacBook-Air-2.local":
    MASTER_FOLDER = Path('..')
    
DATA_FOLDER = MASTER_FOLDER / 'Data'

def get_file(temperature, extrusions, concentration=0.2, lipid="POPC"):
    folder = DATA_FOLDER / lipid / (f'{concentration}_mg_ml') / (f'{extrusions}_extrusions')
    file = folder / (f'{temperature}_degrees')
    return file