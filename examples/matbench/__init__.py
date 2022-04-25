from glob import glob
from os.path import dirname
from typing import Literal

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"

ROOT = __file__.split("examples/matbench")[0]
MODULE_DIR = dirname(__file__)

DATA_PATHS = {
    path.split("/")[-1].split(".")[0]: path
    for path in glob(f"{ROOT}/datasets/matbench_*.json.bz2")
}

assert len(DATA_PATHS) == 13, f"glob found {len(DATA_PATHS)} data sets, expected 13"

MatbenchDatasets = Literal[
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
