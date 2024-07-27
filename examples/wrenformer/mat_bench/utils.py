from __future__ import annotations

import json
from typing import Literal

__author__ = "Janosh Riebesell"
__date__ = "2022-05-10"


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


def merge_json_on_disk(
    dct: dict,
    file_path: str,
    on_non_serializable: Literal["annotate", "error"] = "annotate",
) -> None:
    """Merge a dict into a (possibly) existing JSON file.

    Args:
        file_path (str): Path to JSON file. File will be created if not exist.
        dct (dict): Dictionary to merge into JSON file.
        on_non_serializable ('annotate' | 'error'): What to do with non-serializable
            values encountered in dct. 'annotate' will replace the offending object with
            a string indicating the type, e.g. '<not serializable: function>'. 'error'
            will raise 'TypeError: Object of type function is not JSON serializable'.
            Defaults to 'annotate'.
    """
    try:
        with open(file_path) as json_file:
            data = json.load(json_file, object_hook=_int_keys)

        dct = recursive_dict_merge(data, dct)
    except (FileNotFoundError, json.decoder.JSONDecodeError):  # file missing or empty
        pass

    def non_serializable_handler(obj: object) -> str:
        # replace functions and classes in dct with string indicating it's a
        # non-serializable type
        return f"<not serializable: {type(obj).__qualname__}>"

    with open(file_path, "w") as file:
        default = non_serializable_handler if on_non_serializable == "annotate" else None
        json.dump(dct, file, default=default, indent=2)
