# %%
from __future__ import annotations

import functools
from datetime import datetime
from os.path import basename

from matbench.metadata import mbv01_metadata
from submitit import AutoExecutor

from examples.matbench import DATA_PATHS, MODULE_DIR
from examples.matbench.run_matbench import run_matbench_task

__author__ = "Janosh Riebesell"
__date__ = "2022-04-12"

# %%
model_name = "wren"

log_dir = f"{MODULE_DIR}/job-logs"
benchmark_path = (
    f"{MODULE_DIR}/benchmarks/{model_name}-{datetime.now():%Y-%m-%d@%H-%M}.json"
)

run_task = functools.partial(run_matbench_task, model_name, benchmark_path)

if "roost" in model_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [k for k, v in mbv01_metadata.items() if v.input_type == "structure"]


# %%
# pass cluster="debug" to run jobs in process e.g. to check for runtime errors before
# slurm submission
executor = AutoExecutor(folder=log_dir)
executor.update_parameters(
    job_name=f"matbench-{model_name}",
    timeout_min=4 * 60,
    slurm_partition=(partition := "ampere"),
    nodes=1,
    gpus_per_node=1,
    slurm_additional_parameters={"account": (account := "LEE-JR769-SL2-GPU")},
    slurm_setup=[  # prepended into sbatch script before actual command
        ". /etc/profile.d/modules.sh",  # source module command
        "module purge",  # clear existing modules
        "module load rhel8/default-amp",  # load ampere partition environment
    ],
    comment=f"Matbench {model_name} on {partition = } billed to {account = }",
    # minimal slurm config above, following kwargs are optional
    stderr_to_stdout=True,  # put errors and normal logs into same file to avoid clutter
)


# %% submit a slurm job array
jobs = executor.map_array(run_task, datasets)

print(
    f"job IDs = {jobs[0].job_id} - {jobs[-1].job_id} submitted at "
    f"{datetime.now():%Y-%m-%d %H-%M}"
)


# %% submit single slurm job
# matbench_dielectric, matbench_mp_e_form
job = executor.submit(run_task, "matbench_jdft2d", epochs=1)  #

print(
    f"job ID = {job.job_id} submitted at {datetime.now():%Y-%m-%d %H-%M}.\n"
    f"Benchmark will be saved to {basename(benchmark_path)}."
)

# %%
run_task("matbench_dielectric", epochs=1)
