from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from itertools import chain, groupby, permutations, product
from operator import itemgetter
from os.path import abspath, dirname, join
from shutil import which
from string import ascii_uppercase, digits
from typing import Literal

from monty.fractions import gcd
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from pyxtal import pyxtal
except ImportError:
    pyxtal = None

module_dir = dirname(abspath(__file__))

with open(join(module_dir, "wyckoff-position-multiplicities.json")) as file:
    # dictionary mapping Wyckoff letters in a given space group to their multiplicity
    wyckoff_multiplicity_dict = json.load(file)

with open(join(module_dir, "wyckoff-position-params.json")) as file:
    param_dict = json.load(file)

with open(join(module_dir, "wyckoff-position-relabelings.json")) as file:
    relab_dict = json.load(file)

relab_dict = {
    spg: [{int(key): line for key, line in val.items()} for val in vals]
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

# Define regex patterns as constants
RE_WYCKOFF_NO_PREFIX = re.compile(r"((?<![0-9])[A-z])")
RE_ELEMENT_NO_SUFFIX = re.compile(r"([A-z](?![0-9]))")
RE_WYCKOFF = re.compile(r"(?<!\d)([a-zA-Z])")
RE_ANONYMOUS = re.compile(r"([A-Z])(?![0-9])")
RE_SUBST_ONE_PREFIX = r"1\g<1>"
RE_SUBST_ONE_SUFFIX = r"\g<1>1"


def split_alpha_numeric(s: str) -> dict[str, list[str]]:
    """Split a string into separate lists of alpha and numeric groups.

    Args:
        s (str): The input string to split.

    Returns:
        dict[str, list[str]]: A dictionary with keys 'alpha' and 'numeric',
                              each containing a list of the respective groups.
    """
    groups = ["".join(g) for _, g in groupby(s, str.isalpha)]
    return {
        "alpha": [g for g in groups if g.isalpha()],
        "numeric": [g for g in groups if g.isnumeric()],
    }


def count_values_for_wyckoff(
    wyckoff: list[str],
    multiplicity: list[str],
    spg: str,
    lookup_dict: dict[str, dict[str, int]],
):
    """Count values from a lookup table and scale by wyckoff multiplicities."""
    return sum(float(n) * lookup_dict[spg][k] for n, k in zip(multiplicity, wyckoff))


def get_aflow_label_from_aflow(
    struct: Structure,
    aflow_executable: str | None = None,
    errors: Literal["raise", "annotate", "ignore"] = "raise",
) -> str:
    """Get Aflow prototype label for a pymatgen Structure. Make sure you're running a
    recent version of the aflow CLI as there's been several breaking changes. This code
    was tested under v3.2.12.

    Install guide: https://aflow.org/install-aflow/#install_aflow
        http://aflow.org/install-aflow/install-aflow.sh -o install-aflow.sh
        chmod 555 install-aflow.sh
        ./install-aflow.sh --slim

    Args:
        struct (Structure): pymatgen Structure
        aflow_executable (str): path to aflow executable. Defaults to which("aflow").
        errors ('raise' | 'annotate' | 'ignore']): How to handle errors. 'raise' and
            'ignore' are self-explanatory. 'annotate' prefixes problematic Aflow labels
            with 'invalid <reason>: '.

    Raises:
        ValueError: if errors='raise' and Wyckoff multiplicities do not add up to
            expected composition.

    Returns:
        str: Aflow prototype label
    """
    if aflow_executable is None:
        aflow_executable = which("aflow")

    if which(aflow_executable or "") is None:
        raise FileNotFoundError(
            "AFLOW could not be found, please specify path to its binary with "
            "aflow_executable='...'"
        )

    cmd = f"{aflow_executable} --prototype --print=json cat".split()

    output = subprocess.run(
        cmd,
        input=struct.to(fmt="poscar"),
        text=True,
        capture_output=True,
        check=True,
    )

    aflow_proto = json.loads(output.stdout)

    aflow_label = aflow_proto["aflow_prototype_label"]

    # check that multiplicities satisfy original composition
    _, _, spg_num, *wyckoff_letters = aflow_label.split("_")
    elements = sorted(el.symbol for el in struct.composition)
    elem_dict = {}
    for elem, wyk_letters_per_elem in zip(elements, wyckoff_letters):
        # normalize Wyckoff letters to start with 1 if missing digit
        wyk_letters_normalized = re.sub(
            RE_WYCKOFF_NO_PREFIX, RE_SUBST_ONE_PREFIX, wyk_letters_per_elem
        )
        sep_el_wyks = split_alpha_numeric(wyk_letters_normalized)
        elem_dict[elem] = count_values_for_wyckoff(
            sep_el_wyks["alpha"],
            sep_el_wyks["numeric"],
            spg_num,
            wyckoff_multiplicity_dict,
        )

    full_label = f"{aflow_label}:{'-'.join(elements)}"

    observed_formula = Composition(elem_dict).reduced_formula
    expected_formula = struct.composition.reduced_formula
    if observed_formula != expected_formula:
        if errors == "raise":
            raise ValueError(
                f"invalid WP multiplicities - {aflow_label}, expected "
                f"{observed_formula} to be {expected_formula}"
            )
        if errors == "annotate":
            return f"invalid multiplicities: {full_label}"

    return full_label


def get_aflow_label_from_spglib(
    struct: Structure,
    errors: Literal["raise", "annotate", "ignore"] = "ignore",
    init_symprec: float = 0.1,
    fallback_symprec: float = 1e-5,
) -> str | None:
    """Get AFLOW prototype label for pymatgen Structure.

    Args:
        struct (Structure): pymatgen Structure object.
        errors ('raise' | 'annotate' | 'ignore']): How to handle errors. 'raise' and
            'ignore' are self-explanatory. 'annotate' prefixes problematic Aflow labels
            with 'invalid <reason>: '.
        init_symprec (float): Initial symmetry precision for spglib. Defaults to 0.1.
        fallback_symprec (float): Fallback symmetry precision for spglib if first
            symmetry detection failed. Defaults to 1e-5.

    Returns:
        str: AFLOW prototype label or None if errors='ignore' and symmetry detection
            failed.
    """
    try:
        spg_analyzer = SpacegroupAnalyzer(
            struct, symprec=init_symprec, angle_tolerance=5
        )
        aflow_label_with_chemsys = get_aflow_label_from_spg_analyzer(
            spg_analyzer, errors
        )

        # try again with refined structure if it initially fails
        # NOTE structures with magmoms fail unless all have same magnetic moment
        if "invalid" in aflow_label_with_chemsys:
            spg_analyzer = SpacegroupAnalyzer(
                spg_analyzer.get_refined_structure(),
                symprec=fallback_symprec,
                angle_tolerance=-1,
            )
            aflow_label_with_chemsys = get_aflow_label_from_spg_analyzer(
                spg_analyzer, errors
            )
        return aflow_label_with_chemsys

    except ValueError as exc:
        if errors == "annotate":
            return f"invalid spglib: {exc}"
        raise  # we only get here if errors == "raise"


def get_aflow_label_from_spg_analyzer(
    spg_analyzer: SpacegroupAnalyzer,
    errors: Literal["raise", "annotate", "ignore"] = "raise",
) -> str:
    """Get AFLOW prototype label for pymatgen SpacegroupAnalyzer.

    Args:
        spg_analyzer (SpacegroupAnalyzer): pymatgen SpacegroupAnalyzer object.
        errors ('raise' | 'annotate' | 'ignore']): How to handle errors. 'raise' and
            'ignore' are self-explanatory. 'annotate' prefixes problematic Aflow labels
            with 'invalid <reason>: '.

    Raises:
        ValueError: if errors='raise' and Wyckoff multiplicities do not add up to
            expected composition.

    Raises:
        ValueError: if Wyckoff multiplicities do not add up to expected composition.

    Returns:
        str: AFLOW prototype labels
    """
    spg_num = spg_analyzer.get_space_group_number()
    sym_struct = spg_analyzer.get_symmetrized_structure()

    equivalent_wyckoff_labels = [
        (len(s), s[0].species_string, wyk_letter.translate(remove_digits))
        for s, wyk_letter in zip(
            sym_struct.equivalent_sites, sym_struct.wyckoff_symbols
        )
    ]
    equivalent_wyckoff_labels = sorted(
        equivalent_wyckoff_labels, key=lambda x: (x[1], x[2])
    )

    # check that multiplicities satisfy original composition
    elem_dict = {}
    elem_wyks = []
    for el, g in groupby(
        equivalent_wyckoff_labels, key=lambda x: x[1]
    ):  # sort alphabetically by element
        lg = list(g)
        elem_dict[el] = sum(
            float(wyckoff_multiplicity_dict[str(spg_num)][e[2]]) for e in lg
        )
        wyks = ""
        for wyk, w in groupby(
            lg, key=lambda x: x[2]
        ):  # sort alphabetically by wyckoff letter
            lw = list(w)
            wyks += f"{len(lw)}{wyk}"
        elem_wyks.append(wyks)

    # canonicalize the possible wyckoff letter sequences
    canonical = canonicalize_elem_wyks("_".join(elem_wyks), spg_num)

    # get Pearson symbol
    cry_sys = spg_analyzer.get_crystal_system()
    spg_sym = spg_analyzer.get_space_group_symbol()
    centering = "C" if spg_sym[0] in ("A", "B", "C", "S") else spg_sym[0]
    num_sites_conventional = len(spg_analyzer.get_symmetry_dataset()["std_types"])
    pearson_symbol = f"{cry_sys_dict[cry_sys]}{centering}{num_sites_conventional}"

    prototype_form = prototype_formula(sym_struct.composition)

    chem_sys = sym_struct.composition.chemical_system
    aflow_label_with_chemsys = (
        f"{prototype_form}_{pearson_symbol}_{spg_num}_{canonical}:{chem_sys}"
    )

    observed_formula = Composition(elem_dict).reduced_formula
    expected_formula = sym_struct.composition.reduced_formula
    if observed_formula != expected_formula:
        if errors == "raise":
            raise ValueError(
                f"Invalid WP multiplicities - {aflow_label_with_chemsys}, expected "
                f"{observed_formula} to be {expected_formula}"
            )
        if errors == "annotate":
            return f"invalid multiplicities: {aflow_label_with_chemsys}"

    return aflow_label_with_chemsys


def canonicalize_elem_wyks(elem_wyks: str, spg_num: int | str) -> str:
    """Given an element ordering, canonicalize the associated Wyckoff positions
    based on the alphabetical weight of equivalent choices of origin.

    Args:
        elem_wyks (str): Wren Wyckoff string encoding element types at Wyckoff positions
        spg_num (int | str): International space group number.

    Returns:
        str: Canonicalized Wren Wyckoff encoding.
    """
    isopointal = []

    for trans in relab_dict[str(spg_num)]:
        t = str.maketrans(trans)
        isopointal.append(elem_wyks.translate(t))

    isopointal = list(set(isopointal))

    scores = []
    sorted_iso = []
    for wyks in isopointal:
        sorted_el_wyks, score = sort_and_score_wyks(wyks)
        scores.append(score)
        sorted_iso.append(sorted_el_wyks)

    return sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]


def sort_and_score_wyks(wyks: str) -> tuple[str, int]:
    """Determines the order or Wyckoff positions when canonicalizing Aflow labels.

    Args:
        wyks (str): Wyckoff position substring from AFLOW-style prototype label

    Returns:
        tuple: containing
        - str: sorted Wyckoff position substring for AFLOW-style prototype label
        - int: integer score to rank order when canonicalizing
    """
    score = 0
    sorted_el_wyks = []
    for el_wyks in wyks.split("_"):
        sep_el_wyks = split_alpha_numeric(el_wyks)
        sorted_el_wyks.append(
            "".join(
                [
                    f"{mult}{wyk}" if mult != "1" else wyk
                    for mult, wyk in sorted(
                        zip(sep_el_wyks["numeric"], sep_el_wyks["alpha"]),
                        key=lambda x: x[1],
                    )
                ]
            )
        )
        score += sum(0 if el == "A" else ord(el) - 96 for el in sep_el_wyks["alpha"])

    return "_".join(sorted_el_wyks), score


def prototype_formula(composition: Composition) -> str:
    """An anonymized formula. Unique species are arranged in alphabetical order
    and assigned ascending alphabets. This format is used in the aflow structure
    prototype labelling scheme.

    Args:
        composition (Composition): Pymatgen Composition to process

    Returns:
        str: anonymized formula where the species are in alphabetical order
    """
    reduced = composition.element_composition
    if all(x == int(x) for x in composition.values()):
        reduced /= gcd(*(int(i) for i in composition.values()))

    amounts = [reduced[key] for key in sorted(reduced, key=str)]

    anon = ""
    for e, amt in zip(ascii_uppercase, amounts):
        if amt == 1:
            amt_str = ""
        elif abs(amt % 1) < 1e-8:
            amt_str = str(int(amt))
        else:
            amt_str = str(amt)
        anon += f"{e}{amt_str}"
    return anon


def get_anom_formula_from_prototype_formula(prototype_formula: str) -> str:
    """Get an anonymous formula from a prototype formula."""
    prototype_formula = re.sub(
        RE_ELEMENT_NO_SUFFIX, RE_SUBST_ONE_SUFFIX, prototype_formula
    )
    anom_list = split_alpha_numeric(prototype_formula)

    return "".join(
        [
            f"{el}{num}" if num != "1" else el
            for el, num in zip(
                anom_list["alpha"],
                sorted(anom_list["numeric"]),
            )
        ]
    )


def count_wyckoff_positions(aflow_label: str) -> int:
    """Count number of Wyckoff positions in Wyckoff representation.

    Args:
        aflow_label (str): AFLOW-style prototype label with appended chemical system

    Returns:
        int: number of distinct Wyckoff positions
    """
    aflow_label, _ = aflow_label.split(":")  # remove chemical system
    # discard prototype formula and spg symbol and spg number
    wyk_letters = aflow_label.split("_", maxsplit=3)[-1]
    # throw Wyckoff positions for all elements together
    wyk_letters = wyk_letters.replace("_", "")
    wyk_list = re.split("[A-z]", wyk_letters)[:-1]  # split on every letter

    # count 1 for letters without prefix
    return sum(1 if len(x) == 0 else int(x) for x in wyk_list)


def count_crystal_dof(aflow_label: str) -> int:
    """Count number of free parameters coarse-grained in Wyckoff representation: how
    many degrees of freedom would remain to optimize during a crystal structure
    relaxation.

    Args:
        aflow_label (str): AFLOW-style prototype label with appended chemical system

    Returns:
        int: Number of free-parameters in given prototype
    """
    n_params = 0

    aflow_label, _ = aflow_label.split(":")  # chop off chemical system
    _, pearson, spg, *wyks = aflow_label.split("_")

    n_params += cry_param_dict[pearson[0]]

    for wyk_letters_per_elem in wyks:
        # normalize Wyckoff letters to start with 1 if missing digit
        wyk_letters_normalized = re.sub(
            RE_WYCKOFF_NO_PREFIX, RE_SUBST_ONE_PREFIX, wyk_letters_per_elem
        )
        sep_el_wyks = split_alpha_numeric(wyk_letters_normalized)
        n_params += count_values_for_wyckoff(
            sep_el_wyks["alpha"],
            sep_el_wyks["numeric"],
            spg,
            param_dict,
        )

    return n_params


def get_isopointal_proto_from_aflow(aflow_label: str) -> str:
    """Get a canonicalized string for the prototype.

    Args:
        aflow_label (str): AFLOW-style prototype label with appended chemical system

    Returns:
        str: Canonicalized AFLOW-style prototype label with appended chemical system
    """
    aflow_label, _ = aflow_label.split(":")
    anonymous_formula, pearson, spg, *wyckoffs = aflow_label.split("_")

    anonymous_formula = re.sub(
        RE_ELEMENT_NO_SUFFIX, RE_SUBST_ONE_SUFFIX, anonymous_formula
    )
    anom_list = split_alpha_numeric(anonymous_formula)
    counts = [int(x) for x in anom_list["numeric"]]
    dummy_els = anom_list["alpha"]

    s_counts, s_wyks_tup = list(zip(*sorted(zip(counts, wyckoffs))))
    s_wyks = re.sub(RE_WYCKOFF_NO_PREFIX, RE_SUBST_ONE_PREFIX, "_".join(s_wyks_tup))
    c_anom = "".join(
        [f"{el}{num}" if num != 1 else el for el, num in zip(dummy_els, s_counts)]
    )

    if len(s_counts) == len(set(s_counts)):
        cs_wyks = canonicalize_elem_wyks(s_wyks, int(spg))
        return f"{c_anom}_{pearson}_{spg}_{cs_wyks}"

    # credit Stef: https://stackoverflow.com/a/70126643/5517459
    valid_permutations = [
        list(map(itemgetter(1), chain.from_iterable(p)))
        for p in product(
            *[
                permutations(g)
                for _, g in groupby(
                    sorted(zip(s_counts, s_wyks.split("_"))), key=lambda x: x[0]
                )
            ]
        )
    ]

    isopointal: list[str] = []

    for wyks_list in valid_permutations:
        for trans in relab_dict[spg]:
            t = str.maketrans(trans)
            isopointal.append("_".join(wyks_list).translate(t))

    isopointal = list(set(isopointal))

    scores = []
    sorted_iso = []
    for wyks in isopointal:
        sorted_el_wyks, score = sort_and_score_wyks(wyks)
        scores.append(score)
        sorted_iso.append(sorted_el_wyks)

    canonical = sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))

    # TODO: how to tie break when the scores are the same?
    # currently done by alphabetical
    return "_".join((c_anom, pearson, spg, canonical[0][1]))


def _get_anom_formula_dict(anonymous_formula: str) -> dict:
    """Get a dictionary of element to count from an anonymous formula."""
    result: defaultdict = defaultdict(int)
    element = ""
    count = ""

    for char in anonymous_formula:
        if char.isalpha():
            if element:
                result[element] += int(count) if count else 1
                count = ""
            element = char
        else:
            count += char

    if element:
        result[element] += int(count) if count else 1

    return dict(result)


def _find_translations(
    dict1: dict[str, int], dict2: dict[str, int]
) -> list[dict[str, str]]:
    """Find all possible translations between two dictionaries."""
    if Counter(dict1.values()) != Counter(dict2.values()):
        return []

    keys2 = list(dict2.keys())
    used = set()

    def backtrack(translation, index):
        if index == len(dict1):
            return [translation.copy()]

        key1 = list(dict1.keys())[index]
        value1 = dict1[key1]
        valid_translations = []

        for key2 in keys2:
            if key2 not in used and dict2[key2] == value1:
                used.add(key2)
                translation[key1] = key2
                valid_translations.extend(backtrack(translation, index + 1))
                used.remove(key2)
                del translation[key1]

        return valid_translations

    return backtrack({}, 0)


def get_aflow_strs_from_iso_and_composition(
    isopointal_proto: str, composition: Composition
) -> list[str]:
    """Get a canonicalized string for the prototype.

    Args:
        isopointal_proto (str): AFLOW-style Canonicalized prototype label
        composition (Composition): pymatgen Composition object

    Returns:
        list[str]: List of possible AFLOW-style prototype labels with appended
            chemical systems that can be generated from combinations of the
            input isopointal_proto and composition.
    """
    if not isinstance(isopointal_proto, str):
        raise TypeError(
            f"Invalid isopointal_proto: {isopointal_proto} ({type(isopointal_proto)})"
        )

    anonymous_formula, pearson, spg, *wyckoffs = isopointal_proto.split("_")

    ele_amt_dict = composition.get_el_amt_dict()
    proto_formula = prototype_formula(composition)
    anom_amt_dict = _get_anom_formula_dict(anonymous_formula)

    translations = _find_translations(ele_amt_dict, anom_amt_dict)
    anom_ele_to_wyk = dict(zip(anom_amt_dict.keys(), wyckoffs))
    anonymous_formula = RE_ANONYMOUS.sub(RE_SUBST_ONE_PREFIX, anonymous_formula)

    result = []
    for t in translations:
        wyckoff_part = "_".join(
            RE_WYCKOFF.sub(RE_SUBST_ONE_PREFIX, anom_ele_to_wyk[t[elem]])
            for elem in sorted(t.keys())
        )
        canonicalized_wyckoff = canonicalize_elem_wyks(wyckoff_part, spg)
        chemical_system = "-".join(sorted(t.keys()))

        aflow_str = (
            f"{proto_formula}_{pearson}_{spg}_{canonicalized_wyckoff}:{chemical_system}"
        )
        result.append(aflow_str)

    return result


def count_distinct_wyckoff_letters(aflow_str: str) -> int:
    """Count number of distinct Wyckoff letters in Wyckoff representation."""
    aflow_str, _ = aflow_str.split(":")  # drop chemical system
    _, _, _, wyckoff_letters = aflow_str.split("_", 3)  # drop prototype, Pearson, spg
    wyckoff_letters = wyckoff_letters.translate(remove_digits).replace("_", "")
    return len(set(wyckoff_letters))  # number of distinct Wyckoff letters


def get_random_structure_for_protostructure(protostructure: str, **kwargs) -> Structure:
    """Generate a random structure for a given prototype structure.

    NOTE that due to the random nature of the generation, the output structure
    may be higher symmetry than the requested prototype structure.
    """
    if pyxtal is None:
        raise ImportError("pyxtal is required for this function")

    aflow_label, chemsys = protostructure.split(":")
    _, _, spg, *wyckoffs = aflow_label.split("_")

    wyckoffs = [re.sub(RE_WYCKOFF_NO_PREFIX, RE_SUBST_ONE_PREFIX, w) for w in wyckoffs]
    sep_el_wyks = [split_alpha_numeric(w) for w in wyckoffs]

    species_sites = [
        [
            site
            for m, w in zip(d["numeric"], d["alpha"])
            for site in [f"{wyckoff_multiplicity_dict[spg][w]}{w}"] * int(m)
        ]
        for d in sep_el_wyks
    ]

    species_counts = [
        sum(
            int(wyckoff_multiplicity_dict[spg][w]) * int(m)
            for m, w in zip(d["numeric"], d["alpha"])
        )
        for d in sep_el_wyks
    ]

    p = pyxtal()
    p.from_random(
        dim=3,
        group=int(spg),
        species=chemsys.split("-"),
        numIons=species_counts,
        sites=species_sites,
        **kwargs,
    )
    return p.to_pymatgen()
