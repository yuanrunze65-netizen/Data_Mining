"""Paths and feature utilities shared by the two image experiments."""

import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "DM_2026_Dataset"
WORK_DIR = PROJECT_ROOT / "_work"
RESULTS_DIR = PROJECT_ROOT / "results"


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def scaled_block(values):
    return normalize(np.sqrt(np.maximum(values, 0)))
