# %%
import os

import pandas as pd
import pymatviz as pmv
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

import aviary.wren.utils as wren_utils
from aviary import ROOT
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
df_perov = pd.read_json(DATA_PATHS["matbench_perovskites"]).set_index("mbid")
df_perov = df_perov.rename(columns={"wyckoff": "spglib_wyckoff"})
df_perov["structure"] = df_perov.structure.map(Structure.from_dict)


# %%
# takes ~6h (when running uninterrupted)
for idx, struct in tqdm(df_perov.structure.items(), total=len(df_perov)):
    if pd.isna(df_perov.aflow_wyckoff[idx]):
        df_perov.loc[idx, "aflow_wyckoff"] = (
            wren_utils.get_protostructure_label_from_aflow(
                struct, "/Users/janosh/bin/aflow"
            )
        )


# %%
# takes ~30 sec
for struct in tqdm(df_perov.structure, total=len(df_perov)):
    wren_utils.get_protostructure_label_from_spglib(struct)


# %%
df_perov.dropna().query("wyckoff != aflow_wyckoff")


# %%
print(
    "Percentage of materials with spglib label != aflow label: "
    f"{len(df_perov.query('wyckoff != aflow_wyckoff')) / len(df_perov):.0%}"
)


# %%
df_perov.drop("structure", axis=1).to_csv(
    f"{ROOT}/datasets/matbench_perovskites_protostructure_labels.csv"
)


# %%
df_perov = pd.read_csv(
    f"{ROOT}/datasets/matbench_perovskites_protostructure_labels.csv"
).set_index("mbid")


# %%
for src in ("aflow", "spglib"):
    df_perov[f"{src}_spg_num"] = (
        df_perov[f"{src}_wyckoff"].str.split("_").str[2].astype(int)
    )


# %%
fig = pmv.spacegroup_sunburst(df_perov.spglib_spg)
fig.update_layout(title=dict(text="Spglib Spacegroups", x=0.5, y=0.93))
# fig.write_image(f"{MODULE_DIR}/plots/matbench_perovskites_aflow_sunburst.pdf")


# %%
fig = pmv.spacegroup_sunburst(df_perov.aflow_spg, title="Aflow")
fig.update_layout(title=dict(text="Aflow Spacegroups", x=0.5, y=0.85))
# fig.write_image(f"{MODULE_DIR}/plots/matbench_perovskites_spglib_sunburst.pdf")


# %%
df_perov = load_dataset("matbench_perovskites")

df_perov["spglib_spg_num"] = df_perov.structure.map(
    lambda struct: SpacegroupAnalyzer(struct).get_space_group_number()
)


# %%
for src in ("aflow", "spglib"):
    df_perov[f"{src}_crys_sys"] = df_perov[f"{src}_spg_num"].map(
        pmv.utils.crystal_sys_from_spg_num
    )


# %%
fig = pmv.sankey_from_2_df_cols(df_perov, ["aflow_spg_num", "spglib_spg_num"])

fig.update_layout(title="Matbench Perovskites Aflow vs Spglib Spacegroups")


# %%
fig = pmv.sankey_from_2_df_cols(df_perov, ["aflow_crys_sys", "spglib_crys_sys"])

fig.update_layout(title="Aflow vs Spglib Crystal Systems")
