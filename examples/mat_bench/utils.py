import gzip
import json
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator


def _int_keys(d):
    # JSON annoyingly stringifies all dict keys during serialization and does not revert
    # floats and ints back during parsing. This json.load() hook converts key strings
    # containing only digits to ints.
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()}


@contextmanager
def open_json(filepath: str, initial: dict = None) -> Generator[dict, None, None]:
    """Open a JSON file and yield a 2-level defaultdict of its contents. Save the modified
    dict back to disk on exiting the context manager.

    2-level defaultdict means you can assign d[key1][key2][key3] = value without setting
    d[key1][key2] = {} first.

    Args:
        filepath (str): JSON file path.
        initial (dict, optional): Initial value (not lambda) for defaultdict if JSON file
            was missing or empty. Defaults to None.

    Yields:
        Generator[dict[Any, Any], None, None]: _description_
    """
    open_fn = gzip.open if filepath.lower().endswith(".gz") else open

    try:
        with open_fn(filepath) as json_file:
            loaded = json.load(json_file, object_hook=_int_keys)
            json_data = defaultdict(lambda: defaultdict(dict), loaded)
    except (FileNotFoundError, json.decoder.JSONDecodeError):  # file missing or empty
        json_data = defaultdict(lambda: defaultdict(dict), initial or {})

    try:
        yield json_data
    finally:
        # if user raises exception in with block, still try to save any changes they may
        # have made to data back to disk
        with open_fn(filepath, "wt") as json_file:
            json.dump(json_data, json_file)
