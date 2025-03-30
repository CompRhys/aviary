import os
from glob import glob
from typing import Literal

from matbench.data_ops import load
from pymatgen.analysis.prototypes import get_protostructure_label_from_spglib
from tqdm import tqdm

tqdm.pandas()

current_dir = os.path.dirname(os.path.abspath(__file__))

SMOKE_TEST = True

matbench_datasets = [
    "matbench_steels",
    "matbench_jdft2d",
    "matbench_phonons",
    "matbench_expt_gap",
    "matbench_dielectric",
    "matbench_expt_is_metal",
    "matbench_glass",
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_perovskites",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_mp_e_form",
]

if SMOKE_TEST:
    matbench_datasets = matbench_datasets[2:3]

MatbenchDatasets = Literal[*matbench_datasets]

os.makedirs(f"{current_dir}/datasets", exist_ok=True)
for dataset in matbench_datasets:
    dataset_path = f"{current_dir}/datasets/{dataset}.json.bz2"

    if os.path.exists(dataset_path):
        print(f"Dataset {dataset} already exists, skipping")
        continue

    df = load(dataset)

    if "structure" in df:
        df["composition"] = [struct.formula for struct in df.structure]
        df["protostructure"] = df["structure"].progress_apply(
            get_protostructure_label_from_spglib
        )
    else:
        raise ValueError("No structure or composition column found")

    df.to_json(
        dataset_path,
        default_handler=lambda x: x.as_dict(),
    )


DATA_PATHS = {
    path.split("/")[-1].split(".")[0]: path
    for path in glob(f"{current_dir}/datasets/matbench_*.json.bz2")
}

assert len(DATA_PATHS) == len(
    matbench_datasets
), f"glob found {len(DATA_PATHS)} data sets, expected {len(matbench_datasets)}"
