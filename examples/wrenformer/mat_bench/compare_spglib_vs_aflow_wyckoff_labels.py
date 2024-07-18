# %%
import os

import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatviz import sankey_from_2_df_cols, spacegroup_sunburst
from pymatviz.utils import crystal_sys_from_spg_num
from tqdm import tqdm

from aviary import ROOT
from aviary.wren.utils import (
    get_aflow_label_from_aflow,
    get_protostructure_label_from_spglib,
)
from examples.wrenformer.mat_bench import DATA_PATHS

__author__ = "Janosh Riebesell"
__date__ = "2022-05-17"

"""
This notebook compares the output of Aflow and Spglib algorithms for assigning crystal
symmetries and Wyckoff position to crystal structures. Aflow is much slower but believed
to be more accurate. Materials Project uses Spglib so our spacegroups should be
identical to theirs. CCSD has their own algorithm and we found both Aflow and Spglib
occasionally disagree with their results.
"""

MODULE_DIR = os.path.dirname(__file__)


# %%
df_perovskites = pd.read_json(DATA_PATHS["matbench_perovskites"]).set_index("mbid")
df_perovskites = df_perovskites.rename(columns={"wyckoff": "spglib_wyckoff"})
df_perovskites["structure"] = [
    Structure.from_dict(struct) for struct in df_perovskites.structure
]


# %%
# takes ~6h (when running uninterrupted)
for idx, struct in tqdm(df_perovskites.structure.items(), total=len(df_perovskites)):
    if pd.isna(df_perovskites.aflow_wyckoff[idx]):
        df_perovskites.loc[idx, "aflow_wyckoff"] = get_aflow_label_from_aflow(
            struct, "/Users/janosh/bin/aflow"
        )


# %%
# takes ~30 sec
for struct in tqdm(df_perovskites.structure, total=len(df_perovskites)):
    get_protostructure_label_from_spglib(struct)


# %%
df_perovskites.dropna().query("wyckoff != aflow_wyckoff")


# %%
print(
    "Percentage of materials with spglib label != aflow label: "
    f"{len(df_perovskites.query('wyckoff != aflow_wyckoff')) / len(df_perovskites):.0%}"
)


# %%
# df_perovskites.drop("structure", axis=1).to_csv(
#     f"{ROOT}/datasets/matbench_perovskites_aflow_labels.csv"
# )
df_perovskites = pd.read_csv(
    f"{ROOT}/datasets/matbench_perovskites_aflow_labels.csv"
).set_index("mbid")


# %%
f"{ROOT}/datasets/matbench_perovskites_aflow_labels.csv"


# %%
for src in ("aflow", "spglib"):
    df_perovskites[f"{src}_spg_num"] = (
        df_perovskites[f"{src}_wyckoff"].str.split("_").str[2].astype(int)
    )


# %%
fig = spacegroup_sunburst(df_perovskites.spglib_spg)
fig.update_layout(title=dict(text="Spglib Spacegroups", x=0.5, y=0.93))
# fig.write_image(f"{MODULE_DIR}/plots/matbench_perovskites_aflow_sunburst.pdf")


# %%
fig = spacegroup_sunburst(df_perovskites.aflow_spg, title="Aflow")
fig.update_layout(title=dict(text="Aflow Spacegroups", x=0.5, y=0.85))
# fig.write_image(f"{MODULE_DIR}/plots/matbench_perovskites_spglib_sunburst.pdf")


# %%
df_perovskites = load_dataset("matbench_perovskites")

df_perovskites["spglib_spg_num"] = df_perovskites.structure.map(
    lambda struct: SpacegroupAnalyzer(struct).get_space_group_number()
)


# %%
for src in ("aflow", "spglib"):
    df_perovskites[f"{src}_crys_sys"] = df_perovskites[f"{src}_spg_num"].map(
        crystal_sys_from_spg_num
    )


# %%
fig = sankey_from_2_df_cols(df_perovskites, ["aflow_spg_num", "spglib_spg_num"])

fig.update_layout(title="Matbench Perovskites Aflow vs Spglib Spacegroups")


# %%
fig = sankey_from_2_df_cols(df_perovskites, ["aflow_crys_sys", "spglib_crys_sys"])

fig.update_layout(title="Aflow vs Spglib Crystal Systems")
