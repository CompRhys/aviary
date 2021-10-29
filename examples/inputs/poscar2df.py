# %%
import glob
from functools import partial

import pandas as pd
from pymatgen.core import Composition, Structure
from tqdm.autonotebook import tqdm

from aviary.cgcnn.utils import get_cgcnn_input
from aviary.wren.utils import count_wyks, get_aflow_label_spglib

tqdm.pandas()  # prime progress_apply functionality

final_dir = "data/examples"

idx_list = []
structs = []
E_vasp_list = []
meta_list = []
ht_paths = []

for f in glob.glob(final_dir + "/raw/*.poscar", recursive=True):
    task_id = f.split("/")[-1].split(".")[0]

    with open(f) as s:
        s = s.read()
        struct = Structure.from_str(s, fmt="poscar")

        lines = s.split("\n")

        num = lines[6].split()
        E_vasp_per_atom = float(lines[0].split()[0]) / sum(int(a) for a in num)

        ht_path = lines[0].split()[1]
        meta_data = "[" + lines[0].split("[")[-1]

    idx_list.append(task_id)
    structs.append(struct)
    E_vasp_list.append(E_vasp_per_atom)
    ht_paths.append(ht_path)
    meta_list.append(meta_data)


df = pd.DataFrame()
df["material_id"] = idx_list
df["final_structure"] = structs
df["E_vasp_per_atom"] = E_vasp_list
df["meta_data"] = meta_list
df["ht_data"] = ht_paths

df["E_vasp_per_atom"] = df["E_vasp_per_atom"].astype(float)

print("\n~~~~ LOAD DATA ~~~~")
# Remove duplicated ID's keeping lowest energy
# NOTE this is a bug in TAATA we really shouldn't have to do this
df = df.sort_values(by=["material_id", "E_vasp_per_atom"], ascending=True)
df = df[~df["material_id"].duplicated(keep="first")]

# %%
# Count number of datapoints
print(f"Number of points in dataset: {len(df)}")

symlib = "spglib"  # takes ~ 15mins
df["wyckoff"] = df["final_structure"].progress_apply(get_aflow_label_spglib)

lattice, sites = zip(*df["final_structure"].progress_apply(get_cgcnn_input))

df["composition"] = df["final_structure"].apply(lambda x: x.composition.reduced_formula)
df["nelements"] = df["final_structure"].apply(lambda x: len(x.composition.elements))
df["volume"] = df["final_structure"].apply(lambda x: x.volume)
df["nsites"] = df["final_structure"].apply(lambda x: x.num_sites)

df["lattice"] = lattice
df["sites"] = sites

# %%
# Calculate Formation Enthalpy
df_el = df[df["nelements"] == 1]
df_el = df_el.sort_values(by=["composition", "E_vasp_per_atom"], ascending=True)
el_refs = {
    c.composition.elements[0]: e
    for c, e in zip(df_el["final_structure"], df_el["E_vasp_per_atom"])
}


def get_formation_energy(args, el_refs):
    comp, energy = args
    c = Composition(comp)
    # NOTE our references use energies_per_atom for energy
    ref_e = sum(c[el] * el_refs[el] for el in c.elements)
    return energy - ref_e / c.num_atoms


df["E_f"] = df[["composition", "E_vasp_per_atom"]].apply(
    partial(get_formation_energy, el_refs=el_refs), axis=1
)

# %%
# Remove invalid Wyckoff Sequences
df["nwyckoff"] = df["wyckoff"].apply(count_wyks)

df = df[~df["wyckoff"].str.contains("Invalid")]
print(f"Valid Wyckoff representation {len(df)}")


# %%
# Drop duplicated wyckoff representations
df = df.sort_values(by=["wyckoff", "E_vasp_per_atom"], ascending=True)
df_wyk = df.drop_duplicates(["wyckoff"], keep="first")
print(f"Lowest energy unique wyckoff sequences: {len(df_wyk)}")

# %%
# NOTE searching after having dropped wyckoff duplicates will remove
# some scaled duplicates. This value may still contain duplicates.
df_wyk = df_wyk.sort_values(by=["composition", "E_vasp_per_atom"], ascending=True)
df_comp = df_wyk.drop_duplicates("composition", keep="first")
print(f"Lowest energy polymorphs only: {len(df_comp)}")

# %%
# Clean the data

print("\n~~~~ DATA CLEANING ~~~~")
print(f"Total systems: {len(df_wyk)}")

wyk_lim = 16
df_wyk = df_wyk[df_wyk["nwyckoff"] <= wyk_lim]
print(f"Less than {wyk_lim} Wyckoff species in cell: {len(df_wyk)}")

cell_lim = 64
df_wyk = df_wyk[df_wyk["nsites"] <= cell_lim]
print(f"Less than {cell_lim} atoms in cell: {len(df_wyk)}")

vol_lim = 500
df_wyk = df_wyk[df_wyk["volume"] / df_wyk["nsites"] < vol_lim]
print(f"Less than {vol_lim} A^3 per site: {len(df_wyk)}")

fields = ["material_id", "composition", "E_f", "wyckoff", "lattice", "sites"]

df_wyk[fields].to_csv(
    final_dir + "/examples.csv",
    index=False,
)
