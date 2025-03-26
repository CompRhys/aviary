import pytest


def test_utils_import_error():
    with pytest.raises(ImportError) as exc_info:
        from aviary.wren.utils import relab_dict  # noqa: F401

    assert "functionality from aviary.wren.utils has been moved to pymatgen" in str(
        exc_info.value
    )
