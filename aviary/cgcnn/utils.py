from __future__ import annotations

import numpy as np
from pymatgen.core import Structure


def get_cgcnn_input(struct: Structure) -> tuple[np.ndarray, list[str]]:
    """return a CSV friendly encoding of lattice and sites given a pymatgen structure.

    Args:
        struct (Structure): Pymatgen structure to get inputs for.

    Returns:
        tuple[np.ndarray, list[str]]: CGCCN inputs as a lattice matrix and list of sites
            of the form [f"{el} @ {x,y,z}", ...]
    """
    elems = [atom.specie.symbol for atom in struct]
    cell = struct.lattice.matrix.tolist()
    coords = struct.frac_coords
    sites = [" @ ".join((el, " ".join(map(str, x)))) for el, x in zip(elems, coords)]

    return cell, sites
