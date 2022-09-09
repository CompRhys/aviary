# %%
import os
import subprocess
from datetime import datetime

from matbench.metadata import mbv01_metadata

from examples.wrenformer.mat_bench import DATA_PATHS

__author__ = "Janosh Riebesell, Rokas Elijosius"
__date__ = "2022-04-25"


# %% write Python submission file and sbatch it
epochs = 300
n_attn_layers = 3
embedding_aggregations = ("mean", "std", "min", "max")
folds = list(range(5))
checkpoint = None  # None | 'local' | 'wandb'
lr = 1e-4
model_name = f"wrenformer-robust-{epochs=}"

if "roost" in model_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [k for k, v in mbv01_metadata.items() if v.input_type == "structure"]

log_dir = f"{os.path.dirname(__file__)}/job-logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M-%S}"

python_script = f"""import os
from datetime import datetime
from itertools import product

from examples.wrenformer.mat_bench.train_wrenformer import train_wrenformer_on_matbench

print(f"Job started running {{datetime.now():%Y-%m-%d@%H-%M}}")
job_id = os.environ["SLURM_JOB_ID"]
print(f"{{job_id=}}")
print("{model_name=}")

job_array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print(f"{{job_array_id=}}")

dataset_name, fold = list(product({datasets}, {folds}))[job_array_id]
print(f"{{dataset_name=}}\\n{{fold=}}")

train_wrenformer_on_matbench(
    {model_name=},
    dataset_name=dataset_name,
    {timestamp=},
    fold=fold,
    {epochs=},
    {n_attn_layers=},
    {checkpoint=},
    learning_rate={lr},
    {embedding_aggregations=},
)
"""


submit_script = f"{log_dir}/{timestamp}-{model_name}.py"

# prepend into sbatch script to source module command and load default env
# for Ampere GPU partition before actual job command
slurm_setup = ". /etc/profile.d/modules.sh; module load rhel8/default-amp;"


# %%
slurm_cmd = f"""sbatch
    --partition ampere
    --account LEE-SL3-GPU
    --time 8:0:0
    --nodes 1
    --gpus-per-node 1
    --chdir {log_dir}
    --array 0-{len(datasets) * len(folds) - 1}
    --out {timestamp}-{model_name}-%A-%a.log
    --job-name {model_name}
    --wrap '{slurm_setup} python {submit_script}'
"""

header = f'"""\nSlurm submission command:\n{slurm_cmd}"""'

with open(submit_script, "w") as file:
    file.write(f"{header}\n\n{python_script}")


# %% submit slurm job
subprocess.run(slurm_cmd.replace("\n    ", " "), check=True, shell=True)
print(f"\n{submit_script = }")
