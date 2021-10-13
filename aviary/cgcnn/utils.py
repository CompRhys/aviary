def get_cgcnn_input(struct):
    """return a csv friendly encoding of lattice and sites given a pymatgen structure

    Args:
        struct (Structure): input structure to get inputs for

    Returns:
        cgcnn inputs as a lattice matrix and list of sites of the form [f"{el} @ {x,y,z}", ]
    """
    elems = [atom.specie.symbol for atom in struct]
    cell = struct.lattice.matrix.tolist()
    coords = struct.frac_coords
    sites = [" @ ".join((el, " ".join(map(str, x)))) for el, x in zip(elems, coords)]

    return cell, sites
