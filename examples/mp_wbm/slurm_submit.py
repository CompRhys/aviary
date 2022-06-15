# %%
import os
import subprocess
from datetime import datetime

from aviary import ROOT
from examples.mp_wbm import MODULE_DIR

__author__ = "Janosh Riebesell"
__date__ = "2022-06-13"


# %% write Python submission file and sbatch it
epochs = 300
n_attn_layers = 6
model_name = f"wrenformer-robust-mean+std-aggregation-{epochs=}-{n_attn_layers=}"
fold = 0
n_folds = 1
data_path = f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz"
target = "e_form"
task_type = "regression"
checkpoint = "wandb"  # None | 'local' | 'wandb'
learning_rate = 1e-3

os.makedirs(log_dir := f"{MODULE_DIR}/job-logs", exist_ok=True)
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"

python_script = f"""import os
from datetime import datetime

from examples.mp_wbm.run_wrenformer import run_wrenformer_on_mp_wbm

print(f"Job started running {{datetime.now():%Y-%m-%d@%H-%M}}")
job_id = os.environ["SLURM_JOB_ID"]
print(f"{{job_id=}}")
print("{model_name=}")
print("{data_path=}")

job_array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
print(f"{{job_array_id=}}")

run_wrenformer_on_mp_wbm(
    {model_name=},
    {target=},
    {data_path=},
    {timestamp=},
    test_size=0.05,
    # fold=(n_folds, int(job_array_id)),
    {epochs=},
    {n_attn_layers=},
    {checkpoint=},
    {learning_rate=},
)
"""


submit_script = f"{log_dir}/{timestamp}-{model_name}.py"

# prepend into sbatch script to source module command and load default env
# for Ampere GPU partition before actual job command
slurm_setup = ". /etc/profile.d/modules.sh; module load rhel8/default-amp;"


# %%
slurm_cmd = f"""sbatch
    --partition ampere
    --account LEE-JR769-SL2-GPU
    --time 8:0:0
    --nodes 1
    --gpus-per-node 1
    --chdir {log_dir}
    --array 1-{n_folds}
    --out {timestamp}-{model_name}-%A-%a.log
    --job-name {model_name}
    --wrap '{slurm_setup} python {submit_script}'
"""

header = f'"""\nSlurm submission command:\n{slurm_cmd}"""'

with open(submit_script, "w") as file:
    file.write(f"{header}\n\n{python_script}")


# %% submit slurm job
result = subprocess.run(slurm_cmd.replace("\n    ", " "), check=True, shell=True)
print(f"\n{submit_script = }")
