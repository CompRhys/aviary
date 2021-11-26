import os

from pymatgen.core.structure import Structure

from aviary.wren.utils import (
    count_params,
    count_wyks,
    get_aflow_label_spglib,
    get_isopointal_proto_from_aflow,
)


def test_get_aflow_label_spglib():
    """Check that spglib gives correct aflow input for esseneite
    """
    f = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/ABC6D2_mC40_15_e_e_3f_f.cif"
    )

    with open(f) as s:
        s = s.read()
        struct = Structure.from_str(s, fmt="cif")

    assert get_aflow_label_spglib(struct) == "ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si"


def test_count_wyks():
    """Count the number of wyckoff positions in esseneite
    """
    assert count_wyks("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") == 6


def test_count_params():
    """Count the number of coarse-grained parameters in esseneite
    """
    assert count_params("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") == 18


def test_get_isopointal_proto():
    """Get a recanonicalised prototype string without chemical system
    """
    assert (
        get_isopointal_proto_from_aflow("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si") ==
        "ABC2D6_mC40_15_e_e_f_3f"
    )
    assert (
        get_isopointal_proto_from_aflow("ABC6D2_mC40_15_e_a_3f_f:Ca-Fe-O-Si") ==
        "ABC2D6_mC40_15_a_e_f_3f"
    )
    assert (
        get_isopointal_proto_from_aflow("A_tI8_141_ea:Ca") ==  # not ea is non-canonical
        "A_tI8_141_ae"
    ), "failed to handle single element materials"
    assert (
        get_isopointal_proto_from_aflow("A4BC20D2_oC108_41_2b_a_10b_b:B-Ca-H-N") ==
        "AB2C4D20_oC108_41_a_b_2b_10b"
    ), "failed to reorder elements based on int not first digit"
