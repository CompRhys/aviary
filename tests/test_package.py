import os
from glob import glob

import pytest

from aviary import ROOT

package_sources_path = f"{ROOT}/aviary.egg-info/SOURCES.txt"


__author__ = "Janosh Riebesell"
__date__ = "2022-05-25"


@pytest.mark.skipif(
    not os.path.exists(package_sources_path),
    reason="No aviary.egg-info/SOURCES.txt file, run pip install . to create it",
)
def test_egg_sources():
    """Check we're correctly packaging all JSON files under aviary/ to prevent issues
    like https://github.com/CompRhys/aviary/pull/45.

    This test can fail due to outdated SOURCES.txt. Try `pip install -e .` to update.
    """
    with open(package_sources_path) as file:
        sources = file.read()

    for filepath in glob(f"{ROOT}/aviary/**/*.json", recursive=True):
        rel_path = filepath.split(f"{ROOT}/aviary/")[1]
        assert rel_path in sources
