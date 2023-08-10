import pandas as pd
from matbench import MatbenchBenchmark
from tqdm import tqdm

from aviary import ROOT
from aviary.wren.utils import get_aflow_label_from_spglib

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"


"""
This file uses Spglib to generate Aflow Wyckoff labels for all Matbench datasets and stores them
to disk in the datasets/ folder as Bzip2-compressed JSON files.
"""


benchmark = MatbenchBenchmark()

for idx, task in enumerate(benchmark.tasks, 1):
    print(f"\n\n{idx}/{len(benchmark.tasks)}")
    task.load()
    df: pd.DataFrame = task.df

    if "structure" in df:
        df["composition"] = [struct.formula for struct in df.structure]
        df["wyckoff"] = [
            get_aflow_label_from_spglib(struct)
            for struct in tqdm(df.structure, desc="Getting Aflow Wyckoff labels")
        ]
    elif "composition" in df:
        df["composition"] = [comp.formula for comp in df.composition]
    else:
        raise ValueError("No structure or composition column found")

    df.to_json(
        f"{ROOT}/datasets/{task.dataset_name}.json.bz2",
        default_handler=lambda x: x.as_dict(),
    )
