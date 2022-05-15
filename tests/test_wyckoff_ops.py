import pytest
from pymatgen.core.structure import Structure

from aviary.wren.utils import (
    count_distinct_wyckoff_letters,
    count_params,
    count_wyks,
    get_aflow_label_spglib,
    get_isopointal_proto_from_aflow,
)


def test_get_aflow_label_spglib(tests_dir):
    """Check that spglib gives correct Aflow label for esseneite"""
    struct = Structure.from_file(f"{tests_dir}/data/ABC6D2_mC40_15_e_e_3f_f.cif")

    assert get_aflow_label_spglib(struct) == "ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si"


def test_count_wyks():
    """Count the number of Wyckoff positions in esseneite"""
    assert count_wyks("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") == 6


def test_count_params():
    """Count the number of coarse-grained parameters in esseneite"""
    assert count_params("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") == 18


@pytest.mark.parametrize(
    "aflow_label, expected",
    [
        ("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si", "ABC2D6_mC40_15_e_e_f_3f"),
        ("ABC6D2_mC40_15_e_a_3f_f:Ca-Fe-O-Si", "ABC2D6_mC40_15_a_e_f_3f"),
        # failure of this case means doesn't single element materials
        ("A_tI8_141_ea:Ca", "A_tI8_141_ae"),  # not ea is non-canonical
        # failure of this case means not reordering elements based on int but first digit
        ("A4BC20D2_oC108_41_2b_a_10b_b:B-Ca-H-N", "AB2C4D20_oC108_41_a_b_2b_10b"),
    ],
)
def test_get_isopointal_proto(aflow_label, expected):
    """Get a recanonicalised prototype string without chemical system"""
    assert get_isopointal_proto_from_aflow(aflow_label) == expected


@pytest.mark.parametrize(
    "wyckoff_str, expected",
    [
        ["A20BC14D8E5F2_oP800_61_40c_2c_28c_16c_10c_4c:C-Cd-H-N-O-S", 1],
        ["ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si", 2],
        ["ABC6D2_mC40_15_e_a_3f_f:Ca-Fe-O-Si", 3],
        ["A6B11CD7_aP50_2_6i_ac10i_i_7i:C-H-N-O", 3],
    ],
)
def test_count_distinct_wyckoff_letters(wyckoff_str, expected):
    assert count_distinct_wyckoff_letters(wyckoff_str) == expected
