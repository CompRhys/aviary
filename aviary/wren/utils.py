import json
import re
import subprocess
from itertools import chain, groupby, permutations, product
from operator import itemgetter
from os.path import abspath, dirname, join
from string import ascii_uppercase, digits

from monty.fractions import gcd
from pymatgen.core import Composition, Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

module_dir = dirname(abspath(__file__))

mult_file = join(module_dir, "wp-mult.json")
param_file = join(module_dir, "wp-params.json")
relab_file = join(module_dir, "wp-relab.json")

with open(mult_file) as f:
    mult_dict = json.load(f)

with open(param_file) as f:
    param_dict = json.load(f)

with open(relab_file) as f:
    relab_dict = json.load(f)

relab_dict = {
    spg: [{int(k): l for k, l in val.items()} for val in vals]
    for spg, vals in relab_dict.items()
}

cry_sys_dict = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}

cry_param_dict = {
    "a": 6,
    "m": 4,
    "o": 3,
    "t": 2,
    "h": 2,
    "c": 1,
}

remove_digits = str.maketrans("", "", digits)

AFLOW_EXECUTABLE = "~/bin/aflow"


def get_aflow_label_aflow(
    struct: Structure, aflow_executable: str = AFLOW_EXECUTABLE
) -> str:
    """get aflow prototype label for pymatgen Structure

    args:
        struct (Structure): pymatgen Structure object

    returns:
        aflow prototype label
    """

    poscar = Poscar(struct)

    cmd = f"{aflow_executable} --prototype --print=json cat"

    output = subprocess.run(
        cmd, input=poscar.get_string(), text=True, capture_output=True, shell=True
    )

    aflow_proto = json.loads(output.stdout)

    aflow_label = aflow_proto["aflow_label"]

    # to be consistent with spglib and wren embeddings
    aflow_label = aflow_label.replace("alpha", "A")

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

    aflow_label += ":" + "-".join(elems)

    eqi_comp = Composition(elem_dict)
    if not eqi_comp.reduced_formula == struct.composition.reduced_formula:
        return f"Invalid WP Multiplicities - {aflow_label}"

    return aflow_label


def get_aflow_label_spglib(struct: Structure) -> str:
    """get aflow prototype label for pymatgen Structure

    args:
        struct (Structure): pymatgen Structure object

    returns:
        aflow prototype label
    """
    spga = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5)
    aflow = get_aflow_label_from_spga(spga)

    # try again with refined structure if it initially fails
    # NOTE structures with magmoms fail unless all have same magmom
    if "Invalid" in aflow:
        spga = SpacegroupAnalyzer(
            spga.get_refined_structure(), symprec=1e-5, angle_tolerance=-1
        )
        aflow = get_aflow_label_from_spga(spga)

    return aflow


def get_aflow_label_from_spga(spga: SpacegroupAnalyzer) -> str:
    """get aflow prototype label for pymatgen SpacegroupAnalyzer

    Args:
        spga (SpacegroupAnalyzer): pymatgen SpacegroupAnalyzer object

    Returns:
        aflow prototype labels
    """
    spg_no = spga.get_space_group_number()
    sym_struct = spga.get_symmetrized_structure()

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
        for wyk, w in groupby(
            g, key=lambda x: x[2]
        ):  # sort alphabetically by wyckoff letter
            w = list(w)
            wyks += f"{len(w)}{wyk}"
        elem_wyks.append(wyks)

    # canonicalise the possible wyckoff letter sequences
    elem_wyks = "_".join(elem_wyks)
    canonical = canonicalise_elem_wyks(elem_wyks, spg_no)

    # get pearson symbol
    cry_sys = spga.get_crystal_system()
    spg_sym = spga.get_space_group_symbol()
    centering = "C" if spg_sym[0] in ("A", "B", "C", "S") else spg_sym[0]
    n_conv = len(spga._space_group_data["std_types"])
    pearson = f"{cry_sys_dict[cry_sys]}{centering}{n_conv}"

    prototype_form = prototype_formula(spga._structure.composition)

    aflow_label = (
        f"{prototype_form}_{pearson}_{spg_no}_{canonical}:"
        f"{spga._structure.composition.chemical_system}"
    )

    eqi_comp = Composition(elem_dict)
    if not eqi_comp.reduced_formula == spga._structure.composition.reduced_formula:
        return f"Invalid WP Multiplicities - {aflow_label}"

    return aflow_label


def canonicalise_elem_wyks(elem_wyks: str, spg_no: int) -> str:
    """Given an element ordering, canonicalise the associated Wyckoff positions
    based on the alphabetical weight of equivalent choices of origin.

    Args:
        elem_wyks (str): Wren Wyckoff string encoding element types at Wyckoff
            positions
        spg_no (int): International space group number.

    Returns:
        str: Canonicalised Wren Wyckoff encoding.
    """

    isopointal = []

    for trans in relab_dict[str(spg_no)]:
        t = str.maketrans(trans)
        isopointal.append(elem_wyks.translate(t))

    isopointal = list(set(isopointal))

    scores = []
    sorted_iso = []
    for wyks in isopointal:
        sorted_el_wyks, score = sort_and_score_wyks(wyks)
        scores.append(score)
        sorted_iso.append(sorted_el_wyks)

    canonical = sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]

    return canonical


def sort_and_score_wyks(wyks):
    score = 0
    sorted_el_wyks = []
    for el_wyks in wyks.split("_"):
        sep_el_wyks = ["".join(g) for _, g in groupby(el_wyks, str.isalpha)]
        sep_el_wyks = ["" if i == "1" else i for i in sep_el_wyks]
        sorted_el_wyks.append(
            "".join(
                [
                    f"{n}{w}"
                    for n, w in sorted(
                        zip(sep_el_wyks[0::2], sep_el_wyks[1::2]), key=lambda x: x[1],
                    )
                ]
            )
        )
        score += sum(0 if el == "A" else ord(el) - 96 for el in sep_el_wyks[1::2])

    sorted_el_wyks = "_".join(sorted_el_wyks)

    return sorted_el_wyks, score


def prototype_formula(composition: Composition) -> str:
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
        anon += f"{e}{amt_str}"
    return anon


def count_wyks(aflow_label: str) -> int:
    """Count number of Wyckoff positions in Wyckoff representation
    """
    num_wyk = 0

    aflow_label, _ = aflow_label.split(":")
    wyks = aflow_label.split("_")[3:]

    subst = r"1\g<1>"
    for wyk in wyks:
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)
        sep_el_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]
        try:
            num_wyk += sum(int(n) for n in sep_el_wyks[0::2])
        except ValueError:
            print(sep_el_wyks)
            raise

    return num_wyk


def count_params(aflow_label: str) -> int:
    """Count number of parameters coarse-grained in Wyckoff representation
    """
    num_params = 0

    aflow_label, _ = aflow_label.split(":")
    _, pearson, spg, *wyks = aflow_label.split("_")

    num_params += cry_param_dict[pearson[0]]

    subst = r"1\g<1>"
    for wyk in wyks:
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)
        sep_el_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]
        try:
            num_params += sum(
                float(n) * param_dict[spg][k]
                for n, k in zip(sep_el_wyks[0::2], sep_el_wyks[1::2])
            )
        except ValueError:
            print(sep_el_wyks)
            raise

    return int(num_params)


def get_isopointal_proto_from_aflow(aflow: str) -> str:
    """Get a canonicalised string for the prototype
    """
    aflow, _ = aflow.split(":")
    anom, pearson, spg, *wyks = aflow.split("_")

    subst = r"\g<1>1"
    anom = re.sub(r"([A-z](?![0-9]))", subst, anom)
    anom = ["".join(g) for _, g in groupby(anom, str.isalpha)]
    counts = [int(x) for x in anom[1::2]]
    dummy = anom[0::2]

    s_counts, s_wyks = list(zip(*[(x, y) for x, y in sorted(zip(counts, wyks))]))
    subst = r"1\g<1>"
    s_wyks = re.sub(r"((?<![0-9])[a-zA])", subst, "_".join(s_wyks))
    c_anom = "".join([d + str(c) if c != 1 else d for d, c in zip(dummy, s_counts)])

    if len(s_counts) == len(set(s_counts)):
        cs_wyks = canonicalise_elem_wyks(s_wyks, int(spg))
        return "_".join((c_anom, pearson, spg, cs_wyks))
    else:
        s_wyks = s_wyks.split("_")

        # credit Stef: https://stackoverflow.com/a/70126643/5517459
        valid_permutations = [
            list(map(itemgetter(1), chain.from_iterable(p)))
            for p in product(
                *list(
                    permutations(g)
                    for _, g in groupby(sorted(zip(s_counts, s_wyks)), key=itemgetter(0))
                )
            )
        ]

        scores = []
        sorted_iso = []
        for wyks in valid_permutations:
            sorted_el_wyks, score = sort_and_score_wyks("_".join(wyks))
            scores.append(score)
            sorted_iso.append(sorted_el_wyks)

        canonical = sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]

        # TODO: how to tie break when the scores are the same?
        # currently done by alphabetical
        return "_".join((c_anom, pearson, spg, canonical))
