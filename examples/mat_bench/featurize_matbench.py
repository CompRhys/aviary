from __future__ import annotations

import os

import pandas as pd
from matbench import MatbenchBenchmark
from monty.json import MontyEncoder
from tqdm import tqdm

from aviary.wren.utils import get_aflow_label_spglib

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"

os.makedirs(f"{os.path.dirname(__file__)}/datasets", exist_ok=True)

mbbm = MatbenchBenchmark()

for idx, task in enumerate(mbbm.tasks, 1):
    print(f"\n\n{idx}/{len(mbbm.tasks)}")
    task.load()
    df: pd.DataFrame = task.df

    if "structure" in df:
        df["composition"] = [x.formula for x in df.structure]
        df["wyckoff"] = [
            get_aflow_label_spglib(x)
            for x in tqdm(df.structure, desc="Getting Aflow Wyckoff labels")
        ]
    elif "composition" in df:
        df["composition"] = [x.formula for x in df.composition]
    else:
        raise ValueError("No structure or composition column found")

    df.to_json(
        f"datasets/{task.dataset_name}.json.bz2", default_handler=MontyEncoder().encode
    )
