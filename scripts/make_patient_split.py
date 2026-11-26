#!/usr/bin/env python
"""Write a patient-level split file (tile_id, patient_id, split) from image names '<patient>__<scan>.<ext>'.

    python scripts/make_patient_split.py outputs/bambo_png split.csv --val 8 --test 9 --seed 0

Patients, never tiles, are assigned to a split. A trailing '_TEST' in the patient name is ignored.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("images", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--val", type=int, default=8, help="number of validation patients")
parser.add_argument("--test", type=int, default=9, help="number of test patients")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

stems = sorted(p.stem for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
table = pd.DataFrame({"tile_id": stems, "patient_id": [s.split("__")[0].removesuffix("_TEST") for s in stems]})
patients = sorted(table["patient_id"].unique())
order = np.random.default_rng(args.seed).permutation(len(patients))
assigned = {}
for rank, index in enumerate(order):
    assigned[patients[index]] = "test" if rank < args.test else "validation" if rank < args.test + args.val else "train"
table["split"] = table["patient_id"].map(assigned)
table.to_csv(args.out, index=False)
print(table.groupby("split").agg(patients=("patient_id", "nunique"), tiles=("tile_id", "size")))
