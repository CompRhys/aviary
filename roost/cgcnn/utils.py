def get_cgcnn_input(struct):
    """return the wren input string

    TODO update wren to use a more standard convention

    Args:
        struct (Structure): input structure to get inputs for

    Returns:
        cgcnn inputs as a lattice matrix and list of sites of the form [f"{el} @ {x,y,z}", ]
    """
    eles = [atom.specie.symbol for atom in struct]
    cell = struct.lattice.matrix.tolist()
    coords = struct.frac_coords
    sites = [" @ ".join((el, " ".join(map(str, x)))) for el, x in zip(eles, coords)]

    return cell, sites
