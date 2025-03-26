def __getattr__(name):
    raise ImportError(
        "The functionality from aviary.wren.utils has been moved to pymatgen. "
        "Please install pymatgen using 'pip install pymatgen>2025.3.10' to "
        "access these features."
    )
