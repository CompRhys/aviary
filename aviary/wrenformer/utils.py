import json
import time
from contextlib import contextmanager
from typing import Generator


def _int_keys(dct: dict) -> dict:
    # JSON stringifies all dict keys during serialization and does not revert
    # back to floats and ints during parsing. This json.load() hook converts keys
    # containing only digits to ints.
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in dct.items()}


def recursive_dict_merge(d1: dict, d2: dict) -> dict:
    """Merge two dicts recursively."""
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            recursive_dict_merge(d1[key], d2[key])
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

        dct = recursive_dict_merge(data, dct)
    except (FileNotFoundError, json.decoder.JSONDecodeError):  # file missing or empty
        pass

    with open(file_path, "w") as file:
        json.dump(dct, file)


@contextmanager
def print_walltime(
    start_desc: str = "",
    end_desc: str = "",
    newline: bool = True,
) -> Generator[None, None, None]:
    """Context manager and decorator that prints the wall time of its lifetime.

    Args:
        start_desc (str): Text to print when entering context. Defaults to ''.
        end_desc (str): Text to print when exiting context. Will be followed by 'took
            {duration} sec'. i.e. f"{end_desc} took 1.23 sec". Defaults to ''.
        newline (bool): Whether to print a newline after start_desc. Defaults to True.
    """
    start_time = time.perf_counter()
    if start_desc:
        print(start_desc, end="\n" if newline else "")

    try:
        yield
    finally:
        run_time = time.perf_counter() - start_time
        print(f"{end_desc} took {run_time:.2f} sec")
