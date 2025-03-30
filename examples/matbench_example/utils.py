import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence, Sized
from typing import Final, Literal, TypeVar

# taken from https://slurm.schedmd.com/job_array.html#env_vars, lower-cased and
# and removed the SLURM_ prefix
SLURM_KEYS: Final[tuple[str, ...]] = (
    "job_id",
    "array_job_id",
    "array_task_id",
    "array_task_count",
    "mem_per_node",
    "nodelistsubmit_host",
    "job_partition",
    "job_user",
    "job_account",
    "tasks_per_node",
    "job_qos",
)
SLURM_SUBMIT_KEY: Final[str] = "slurm-submit"
HasLen = TypeVar("HasLen", bound=Sized)


def _int_keys(dct: dict) -> dict:
    # JSON stringifies all dict keys during serialization and does not revert
    # back to floats and ints during parsing. This json.load() hook converts keys
    # containing only digits to ints.
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in dct.items()}


def recursive_dict_merge(dict1: dict, dict2: dict) -> dict:
    """Merge two dicts recursively."""
    for key, val2 in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(val2, dict):
            recursive_dict_merge(dict1[key], val2)
        else:
            dict1[key] = val2
    return dict1


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


def _get_calling_file_path(frame: int = 1) -> str:
    """Return calling file's path.

    Args:
        frame (int, optional): How many function call's up? Defaults to 1.

    Returns:
        str: Calling function's file path n frames up the stack.
    """
    caller_path = sys._getframe(frame).f_code.co_filename
    return os.path.abspath(caller_path)


def slurm_submit(
    job_name: str,
    out_dir: str,
    *,
    time: str | None = None,
    account: str | None = None,
    partition: str | None = None,
    py_file_path: str | None = None,
    slurm_flags: str | Sequence[str] = (),
    array: str | None = None,
    pre_cmd: str = "",
    submit_as_temp_file: bool = True,
) -> dict[str, str]:
    """Slurm submits a python script using `sbatch --wrap 'python path/to/file.py'`.

    Usage: Call this function at the top of the script (before doing any real work) and
    then submit a job with `python path/to/that/script.py slurm-submit`. The slurm job
    will run the whole script.

    Args:
        job_name (str): Slurm job name.
        out_dir (str): Directory to write slurm logs. Log file will include slurm job
            ID and array task ID.
        time (str): 'HH:MM:SS' time limit for the job.
            Defaults to the path of the file calling slurm_submit().
        account (str): Account to charge for this job.
        partition (str, optional): Slurm partition.
        py_file_path (str, optional): Path to the python script to be submitted.
        slurm_flags (str | list[str], optional): Extra slurm CLI flags. Defaults to ().
            Examples: ('--nodes 1', '--gpus-per-node 1') or ('--mem', '16G').
        array (str, optional): Slurm array specifier. Defaults to None. Example:
            '9' (for SLURM_ARRAY_TASK_ID from 0-9 inclusive), '1-10' or '1-10%2', etc.
        pre_cmd (str, optional): Things like `module load` commands and environment
            variables to set before running the python script go here. Example:
            pre_cmd='ENV_VAR=42' or 'module load pytorch;'. Defaults to "". If running
            on CPU, pre_cmd="unset OMP_NUM_THREADS" allows PyTorch to use all cores.
        submit_as_temp_file (bool, optional): If True, copy the Python file to a
            temporary directory before submitting. This allows the user to modify
            the original file without affecting queued jobs. Defaults to True.

    Raises:
        SystemExit: Exit code will be subprocess.run(['sbatch', ...]).returncode.

    Returns:
        dict[str, str]: Slurm variables like job ID, array task ID, compute nodes IDs,
            submission node ID and total job memory.
    """
    py_file_path = py_file_path or _get_calling_file_path(frame=2)

    os.makedirs(out_dir, exist_ok=True)  # slurm fails if out_dir is missing

    # Copy the file to a temporary directory if submit_as_temp_file is True
    if submit_as_temp_file and SLURM_SUBMIT_KEY in sys.argv:
        temp_dir = tempfile.mkdtemp(prefix="slurm_job_")
        temp_file_path = f"{temp_dir}/{os.path.basename(py_file_path)}"
        shutil.copy2(py_file_path, temp_file_path)
        py_file_path = temp_file_path

    # ensure pre_cmd ends with a semicolon
    if pre_cmd and not pre_cmd.strip().endswith(";"):
        pre_cmd += ";"

    cmd = [
        *("sbatch", "--job-name", job_name),
        *("--output", f"{out_dir}/slurm-%A{'-%a' if array else ''}.log"),
        *(slurm_flags.split() if isinstance(slurm_flags, str) else slurm_flags),
        *("--wrap", f"{pre_cmd or ''} python {py_file_path}".strip()),
    ]
    for flag in (f"{time=!s}", f"{account=!s}", f"{partition=!s}", f"{array=!s}"):
        key, val = flag.split("=")
        if val != "None":
            cmd += (f"--{key}", val)

    is_log_file = not sys.stdout.isatty()
    is_slurm_job = "SLURM_JOB_ID" in os.environ

    slurm_vars = {
        f"slurm_{key}": os.environ[f"SLURM_{key}".upper()]
        for key in SLURM_KEYS
        if f"SLURM_{key}".upper() in os.environ
    }
    if time is not None:
        slurm_vars["slurm_timelimit"] = time
    if slurm_flags != ():
        slurm_vars["slurm_flags"] = str(slurm_flags)
    if pre_cmd not in ("", None):
        slurm_vars["pre_cmd"] = pre_cmd

    # print sbatch command into slurm log file and at job submission time
    # but not into terminal or Jupyter
    if (is_slurm_job and is_log_file) or SLURM_SUBMIT_KEY in sys.argv:
        print(f"\n{' '.join(cmd)}\n".replace(" --", "\n  --"))
    if is_slurm_job and is_log_file:
        for key, val in slurm_vars.items():
            print(f"{key}={val}")

    if SLURM_SUBMIT_KEY not in sys.argv:
        return slurm_vars  # if not submitting slurm job, resume outside code as normal

    result = subprocess.run(cmd, check=True)

    # after sbatch submission, exit with slurm exit code
    raise SystemExit(result.returncode)
