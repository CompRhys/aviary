import re
import os
import json
import subprocess
from string import digits, ascii_uppercase
from itertools import groupby
from monty.fractions import gcd

from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar


mult_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allowed-wp-mult.json")
relab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relab.json")

with open(mult_file, "r") as f:
    mult_dict = json.load(f)

with open(relab_file, "r") as f:
    relab_dict = json.load(f)

relab_dict = {
    spg: [{int(k): l for k, l in val.items()} for val in vals]
    for spg, vals in relab_dict.items()
}

cry_sys_dict = {
        "triclinic": "a",
        'monoclinic': "m",
        'orthorhombic': "m",
        'tetragonal': "t",
        'trigonal': "h",
        'hexagonal': "h",
        'cubic': "c",
    }

remove_digits = str.maketrans("", "", digits)

AFLOW_EXECUTABLE = "~/bin/aflow"


# %%
def get_aflow_label_aflow(struct, aflow_executable=AFLOW_EXECUTABLE) -> str:
    """get aflow prototype label for pymatgen structure

    args:
        struct (Structure): pymatgen structure object

    returns:
        aflow prototype labels
    """

    poscar = Poscar(struct)

    cmd = f"{aflow_executable} --prototype --print=json cat"

    output = subprocess.run(
        cmd, input=poscar.get_string(), text=True, capture_output=True, shell=True
    )

    aflow_proto = json.loads(output.stdout)

    aflow_label = aflow_proto["aflow_label"]

    aflow_label = aflow_label.replace("alpha", "A")  # to be consistent with spglib and wren embeddings

    # check that multiplicities satisfy original composition
    symm = aflow_label.split("_")
    spg_no = symm[2]
    wyks = symm[3:]
    elems = poscar.site_symbols
    elem_dict = {}
    subst = r"1\g<1>"
    for el, wyk in zip(elems, wyks):
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)
        sep_el_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]
        elem_dict[el] = sum(
            float(mult_dict[spg_no][w]) * float(n)
            for n, w in zip(sep_el_wyks[0::2], sep_el_wyks[1::2])
        )

    aflow_label += ":"+"-".join(elems)

    eqi_comp = Composition(elem_dict)
    if not eqi_comp.reduced_formula == struct.composition.reduced_formula:
        return f"Invalid WP Multiplicities - {aflow_label}"

    return aflow_label


def get_aflow_label_spglib(struct) -> str:
    """get aflow prototype label for pymatgen structure

    args:
        struct (Structure): pymatgen structure object

    returns:
        aflow prototype labels
    """

    spg = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5)
    spg_no = spg.get_space_group_number()
    sym_struct = spg.get_symmetrized_structure()

    equivs = [
        (len(s), s[0].species_string, f"{wyk.translate(remove_digits)}")
        for s, wyk in zip(sym_struct.equivalent_sites, sym_struct.wyckoff_symbols)
    ]
    equivs = sorted(equivs, key=lambda x: (x[1], x[2]))

    # check that multiplicities satisfy original composition
    elem_dict = {}
    elem_wyks = []
    for el, g in groupby(equivs, key=lambda x: x[1]):  # sort alphabetically by element
        g = list(g)
        elem_dict[el] = sum(float(mult_dict[str(spg_no)][e[2]]) for e in g)
        wyks = ""
        for wyk, w in groupby(g, key=lambda x: x[2]):  # sort alphabetically by wyckoff letter
            w = list(w)
            wyks += f"{len(w)}{wyk}"
        elem_wyks.append(wyks)

    # cannonicalise the possible wyckoff letter sequences
    elem_wyks = "_".join(elem_wyks)
    cannonical = cannonicalise_elem_wyks(elem_wyks, spg_no)

    # get pearson symbol
    cry_sys = spg.get_crystal_system()
    spg_sym = spg.get_space_group_symbol()
    centering = "C" if spg_sym[0] in ("A", "B", "C", "S") else spg_sym[0]
    n_conv = len(spg._space_group_data["std_types"])
    pearson = f"{cry_sys_dict[cry_sys]}{centering}{n_conv}"

    prototype_form = prototype_formula(struct.composition)

    aflow_label = (
        f"{prototype_form}_{pearson}_{spg_no}_{cannonical}:{struct.composition.chemical_system}"
    )

    eqi_comp = Composition(elem_dict)
    if not eqi_comp.reduced_formula == struct.composition.reduced_formula:
        return f"Invalid WP Multiplicities - {aflow_label}"

    return aflow_label


def cannonicalise_elem_wyks(elem_wyks, spg_no):
    """
    Given an element ordering cannonicalise the associated wyckoff positions
    based on the alphabetical weight of equivalent choices of origin.
    """

    isopointial = []

    for trans in relab_dict[str(spg_no)]:
        t = str.maketrans(trans)
        isopointial.append(elem_wyks.translate(t))

    isopointial = list(set(isopointial))

    scores = []
    sorted_iso = []
    for wyks in isopointial:
        score = 0
        sorted_el_wyks = []
        for el_wyks in wyks.split("_"):
            sep_el_wyks = ["".join(g) for _, g in groupby(el_wyks, str.isalpha)]
            sep_el_wyks = ["" if i == "1" else i for i in sep_el_wyks]
            sorted_el_wyks.append(
                "".join([
                    f"{n}{w}"
                    for n, w in sorted(
                        zip(sep_el_wyks[0::2], sep_el_wyks[1::2]),
                        key=lambda x: x[1],
                    )]
                )
            )
            score += sum(0 if l == "A" else ord(l) - 96 for l in sep_el_wyks[1::2])

        scores.append(score)
        sorted_iso.append("_".join(sorted_el_wyks))

    cannonical = sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]

    return cannonical


def prototype_formula(composition) -> str:
    """
    An anonymized formula. Unique species are arranged in alphabetical order
    and assigned ascending alphabets. This format is used in the aflow structure
    prototype labelling scheme.
    """
    reduced = composition.element_composition
    if all(x == int(x) for x in composition.values()):
        reduced /= gcd(*(int(i) for i in composition.values()))

    amts = [amt for _, amt in sorted(reduced.items(), key=lambda x: str(x[0]))]

    anon = ""
    for e, amt in zip(ascii_uppercase, amts):
        if amt == 1:
            amt_str = ""
        elif abs(amt % 1) < 1e-8:
            amt_str = str(int(amt))
        else:
            amt_str = str(amt)
        anon += "{}{}".format(e, amt_str)
    return anon


def count_wyks(aflow_label):
    num_wyk = 0

    aflow_label, _ = aflow_label.split(":")
    wyks = aflow_label.split("_")[3:]

    subst = r"1\g<1>"
    for wyk in wyks:
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)
        sep_el_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]
        try:
            num_wyk += sum(float(n) for n in sep_el_wyks[0::2])
        except ValueError:
            print(sep_el_wyks)
            raise

    return num_wyk
