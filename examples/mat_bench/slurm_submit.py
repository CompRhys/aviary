# %%
import os
from datetime import datetime
from glob import glob

from matbench.metadata import mbv01_metadata

from examples.mat_bench import DATA_PATHS, MODULE_DIR

__author__ = "Janosh Riebesell, Rokas Elijosius"
__date__ = "2022-04-25"


# %% write Python submission file and sbatch it
model_name = "wrenformer"
epochs = 100
folds = list(range(5))

if "roost" in model_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [k for k, v in mbv01_metadata.items() if v.input_type == "structure"]

os.makedirs(log_dir := f"{MODULE_DIR}/job-logs", exist_ok=True)
benchmark_path = (
    f"{MODULE_DIR}/benchmarks/{model_name}-{datetime.now():%Y-%m-%d@%H-%M}.json"
)
job_name = f"matbench-{model_name}-{len(datasets)}jobs"

python_cmd = f"""import os
from itertools import product

from examples.mat_bench.run_matbench import run_matbench_task

job_id = os.environ["SLURM_JOB_ID"]
print(f"{{job_id=}}")

job_array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"{{job_array_id=}}")

dataset_name, fold = list(product({datasets}, {folds}))[job_array_id]
print(f"{{dataset_name=}}\\n{{fold=}}")

run_matbench_task(
    {model_name=},
    {benchmark_path=},
    dataset_name=dataset_name,
    fold=fold,
    {epochs=}
)
"""


submit_script = f"{log_dir}/{job_name}-{datetime.now():%Y-%m-%d@%H-%M}.py"

# prepend into sbatch script to source module command and load default env
# for Ampere GPU partition before actual job command
slurm_setup = ". /etc/profile.d/modules.sh; module load rhel8/default-amp;"

# %%
# --array 0-{len(datasets) * len(folds) - 1}
slurm_cmd = f"""sbatch
    --partition ampere
    --account LEE-JR769-SL2-GPU
    --time 4:0:0
    --nodes 1
    --gpus-per-node 1
    --chdir {log_dir}
    --array 0-5
    --out {job_name}-%A-%a.log
    --job-name {job_name}
    --wrap '{slurm_setup} python {submit_script}'
"""

header = f'"""\nSlurm submission command:\n{slurm_cmd}"""'

with open(submit_script, "w") as file:
    file.write(f"{header}\n\n{python_cmd}")


# %% submit slurm job
# !{slurm_cmd.replace("\n    ", " ")}


# %% clean up log files for failed jobs
job_id = "59883449"
n_removed = len(list(map(os.remove, glob(f"{log_dir}/*-{job_id}-*.log"))))

print(f"{n_removed} log files removed for {job_id=}")
