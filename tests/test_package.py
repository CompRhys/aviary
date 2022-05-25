import os
from glob import glob

import pytest

from aviary import ROOT

package_sources_path = f"{ROOT}/aviary.egg-info/SOURCES.txt"


@pytest.mark.skipif(
    not os.path.exists(package_sources_path),
    reason="No aviary.egg-info/SOURCES.txt file, run pip install . to create it",
)
def test_egg_sources():
    with open(package_sources_path) as file:
        sources = file.read()

    json_files_under_aviary = glob(
        "**/*.json", recursive=True, root_dir=f"{ROOT}/aviary"
    )

    for json_file in json_files_under_aviary:
        assert json_file in sources, f"{json_file} not found in SOURCES.txt"
