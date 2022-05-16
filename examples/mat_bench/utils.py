import json
import time
from contextlib import contextmanager
from typing import Generator


def _int_keys(dct: dict) -> dict:
    # JSON annoyingly stringifies all dict keys during serialization and does not revert
    # floats and ints back during parsing. This json.load() hook converts keys
    # containing only digits to ints.
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in dct.items()}


def dict_merge(d1: dict, d2: dict) -> dict:
    """Merge two dicts recursively."""
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            dict_merge(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def merge_json_on_disk(dct: dict, file_path: str) -> None:
    """Merge a dict into a (possibly) existing JSON file.

    Args:
        file_path (str): Path to JSON file. File will be created if not exist.
        dct (dict): Dictionary to merge into JSON file.
    """

    try:
        with open(file_path) as json_file:
            data = json.load(json_file, object_hook=_int_keys)

        dct = dict_merge(data, dct)
    except (FileNotFoundError, json.decoder.JSONDecodeError):  # file missing or empty
        pass

    with open(file_path, "w") as file:
        json.dump(dct, file)


@contextmanager
def print_walltime(desc: str = "Execution") -> Generator[None, None, None]:
    """Context manager and decorator that prints the wall time of its lifetime.

    Args:
        desc (str, optional): Description prints as f"{desc} took 1.23 sec".
            Defaults to "Execution".
    """
    start_time = time.perf_counter()

    try:
        yield
    finally:
        run_time = time.perf_counter() - start_time
        print(f"{desc} took {run_time:.2f} sec")
