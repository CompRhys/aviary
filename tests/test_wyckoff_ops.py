from itertools import permutations
from shutil import which

import pytest
from pymatgen.core.structure import Composition, Structure

from aviary.wren.utils import (
    _find_translations,
    count_crystal_dof,
    count_distinct_wyckoff_letters,
    count_wyckoff_positions,
    get_aflow_label_from_aflow,
    get_aflow_label_from_spglib,
    get_aflow_strs_from_iso_and_composition,
    get_anom_formula_from_prototype_formula,
    get_isopointal_proto_from_aflow,
    prototype_formula,
)

from .conftest import TEST_DIR


def test_get_aflow_label_from_spglib():
    """Check that spglib gives correct Aflow label for esseneite"""
    struct = Structure.from_file(f"{TEST_DIR}/data/ABC6D2_mC40_15_e_e_3f_f.cif")

    assert get_aflow_label_from_spglib(struct) == "ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si"


@pytest.mark.parametrize(
    "aflow_label, count",
    [
        ("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si", 6),  # esseneite
        ("A6B11CD7_aP50_2_6i_ac10i_i_7i:C-H-N-O", 26),
        ("foo_bar_47_abc_A_b:X-Y-Z", 5),
    ],
)
def test_count_wyckoff_positions(aflow_label, count):
    assert count_wyckoff_positions(aflow_label) == count


def test_count_crystal_dof():
    """Count the number of coarse-grained parameters in esseneite"""
    assert count_crystal_dof("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") == 18


@pytest.mark.parametrize(
    "aflow_label, expected",
    [
        ("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si", "ABC2D6_mC40_15_e_e_f_3f"),
        ("ABC6D2_mC40_15_e_a_3f_f:Ca-Fe-O-Si", "ABC2D6_mC40_15_a_e_f_3f"),
        # this case failing means doesn't handle single element materials
        ("A_tI8_141_ea:Ca", "A_tI8_141_ae"),  # not ea is non-canonical
        # this case failing means not reordering elements based on int but first digit
        ("A4BC20D2_oC108_41_2b_a_10b_b:B-Ca-H-N", "AB2C4D20_oC108_41_a_b_2b_10b"),
    ],
)
def test_get_isopointal_proto(aflow_label, expected):
    """Get a recanonicalized prototype string without chemical system"""
    assert get_isopointal_proto_from_aflow(aflow_label) == expected


@pytest.mark.parametrize(
    "isopointal_proto, composition, expected",
    [
        (
            "AB2C3D4_tP10_115_a_g_bg_cdg",
            "Ce2Al3GaPd4",
            "A3B2CD4_tP10_115_ag_g_b_cdg:Al-Ce-Ga-Pd",
        ),
        # checks that we can handle cases where one element could be on multiple sites
        (
            "ABC3_oP20_62_a_c_cd",
            "YbNiO3",
            "AB3C_oP20_62_c_cd_a:Ni-O-Yb AB3C_oP20_62_a_cd_c:Ni-O-Yb",
        ),
    ],
)
def test_get_aflow_strs_from_iso_and_composition(
    isopointal_proto, composition, expected
):
    aflows = get_aflow_strs_from_iso_and_composition(
        isopointal_proto, Composition(composition)
    )
    assert aflows == expected.split(" ")

    # check the round trip
    assert all(
        get_isopointal_proto_from_aflow(aflow) == isopointal_proto for aflow in aflows
    )


@pytest.mark.parametrize(
    "dict1, dict2, expected",
    [
        # Test case 1: Simple valid translation
        ({"a": 1, "b": 2}, {"x": 1, "y": 2}, [{"a": "x", "b": "y"}]),
        # Test case 2: Multiple valid translations
        (
            {"a": 1, "b": 1, "c": 1},
            {"x": 1, "y": 1, "z": 1},
            [
                dict(zip(["a", "b", "c"], perm))
                for perm in permutations(["x", "y", "z"])
            ],
        ),
        # Test case 3: No valid translations (different values)
        ({"a": 1, "b": 2}, {"x": 1, "y": 3}, []),
        # Test case 4: No valid translations (different number of items)
        ({"a": 1, "b": 2}, {"x": 1, "y": 2, "z": 3}, []),
        # Test case 5: Empty dictionaries
        ({}, {}, [{}]),
        # Test case 6: Larger dictionaries
        (
            {"a": 1, "b": 4, "c": 3, "d": 4},
            {"w": 4, "x": 3, "y": 4, "z": 1},
            [
                {"a": "z", "b": "y", "c": "x", "d": "w"},
                {"a": "z", "b": "w", "c": "x", "d": "y"},
            ],
        ),
    ],
)
def test_find_translations(dict1, dict2, expected):
    result = _find_translations(dict1, dict2)
    assert len(result) == len(expected)
    for translation in result:
        assert translation in expected


# Additional test for performance with larger input
def test_find_translations_performance():
    dict1 = {f"key{i}": i for i in range(8)}
    dict2 = {f"val{i}": i for i in range(8)}
    result = _find_translations(dict1, dict2)
    assert len(result) == 1  # There should be only one valid translation


@pytest.mark.parametrize(
    "composition, expected",
    [("Ce2Al3GaPd4", "A3B2CD4"), ("YbNiO3", "AB3C"), ("K2NaAlF6", "AB6C2D")],
)
def test_prototype_formula(composition: str, expected: str):
    assert prototype_formula(Composition(composition)) == expected


@pytest.mark.parametrize(
    "composition, expected",
    [("Ce2Al3GaPd4", "AB2C3D4"), ("YbNiO3", "ABC3"), ("K2NaAlF6", "ABC2D6")],
)
def test_get_anom_formula_from_prototype_formula(composition: str, expected: str):
    assert get_anom_formula_from_prototype_formula("A3B2CD4") == "AB2C3D4"


@pytest.mark.parametrize(
    "aflow_label, expected",
    [
        ("A20BC14D8E5F2_oP800_61_40c_2c_28c_16c_10c_4c:C-Cd-H-N-O-S", 1),
        ("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si", 2),
        ("ABC6D2_mC40_15_e_a_3f_f:Ca-Fe-O-Si", 3),
        ("A6B11CD7_aP50_2_6i_ac10i_i_7i:C-H-N-O", 3),
    ],
)
def test_count_distinct_wyckoff_letters(aflow_label, expected):
    assert count_distinct_wyckoff_letters(aflow_label) == expected


aflow_cli = which("aflow")


@pytest.mark.skipif(aflow_cli is None, reason="aflow CLI not installed")
def test_get_aflow_label_from_aflow():
    """Check we extract corred correct aflow label for esseneite from  Aflow CLI"""
    struct = Structure.from_file(f"{TEST_DIR}/data/ABC6D2_mC40_15_e_e_3f_f.cif")

    out = get_aflow_label_from_aflow(struct, aflow_cli)
    expected = "ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si"
    assert out == expected
