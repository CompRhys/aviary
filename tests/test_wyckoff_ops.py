import os

from pymatgen.core.structure import Structure

from aviary.wren.utils import count_params, count_wyks, get_aflow_label_spglib


def test_get_aflow_label_spglib():
    """Check that spglib gives correct aflow input for esseneite
    """
    f = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/ABC6D2_mC40_15_e_e_3f_f.cif"
    )

    with open(f) as s:
        s = s.read()
        struct = Structure.from_str(s, fmt="cif")

    aflow = get_aflow_label_spglib(struct)
    assert aflow == "ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si"


def test_count_wyks():
    """Count the number of wyckoff positions in esseneite
    """
    count = count_wyks("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si")
    assert count == 6


def test_count_params():
    """Count the number of coarse-grained parameters in esseneite
    """
    count = count_params("ABC6D2_mC40_15_e_e_3f_f:Ca-Fe-O-Si")
    assert count == 18


if __name__ == "__main__":
    test_get_aflow_label_spglib()
    test_count_wyks()
    test_count_params()
