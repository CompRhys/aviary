import os

import pytest
import torch
from matminer.datasets import load_dataset

from aviary.cgcnn.utils import get_cgcnn_input
from aviary.wren.utils import get_aflow_label_spglib

__author__ = "Janosh Riebesell"
__date__ = "2022-04-09"

torch.manual_seed(0)  # ensure reproducible results (applies to all tests)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def df_matbench_phonons():
    """Returns the dataframe for the Matbench phonon DOS peak task."""

    df = load_dataset("matbench_phonons")
    df[["lattice", "sites"]] = [get_cgcnn_input(x) for x in df.structure]
    df["material_id"] = [f"mb_phdos_{i+1}" for i in range(len(df))]
    df = df.set_index("material_id", drop=False)
    df["composition"] = [x.composition.formula.replace(" ", "") for x in df.structure]

    df["phdos_clf"] = [1 if x > 450 else 0 for x in df["last phdos peak"]]

    return df


@pytest.fixture(scope="session")
def df_matbench_jdft2d():
    """Returns the dataframe for the Matbench experimental band gap task."""

    df = load_dataset("matbench_jdft2d")
    df["material_id"] = [f"mb_jdft2d_{i+1}" for i in range(len(df))]
    df = df.set_index("material_id", drop=False)
    df["composition"] = [x.composition.formula.replace(" ", "") for x in df.structure]

    df["wyckoff"] = [get_aflow_label_spglib(x) for x in df.structure]

    return df


@pytest.fixture(scope="session")
def df_matbench_phonons_wyckoff(df_matbench_phonons):
    """Getting Aflow labels is expensive so we split into a separate fixture to avoid
    paying for it unless requested.
    """
    df_matbench_phonons["wyckoff"] = [
        get_aflow_label_spglib(x) for x in df_matbench_phonons.structure
    ]

    return df_matbench_phonons
