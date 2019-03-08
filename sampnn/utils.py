def load_file(file):
    """
    load file and return a list, files should contain one item per line

    example:    LaCu04
                K2MgO4
                NaCl

    """
    with open(file) as f:
        compositions = f.read().splitlines()
    return compositions

