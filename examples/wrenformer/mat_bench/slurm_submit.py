# %%
import os
from datetime import datetime
from itertools import product

from matbench.metadata import mbv01_metadata
from matbench_discovery.slurm import slurm_submit_python

from aviary import ROOT
from examples.wrenformer.mat_bench import DATA_PATHS
from examples.wrenformer.mat_bench.train import train_wrenformer_on_matbench

"""
Train a Wrenformer ensemble of size n_folds on collection of Matbench datasets.
"""

__author__ = "Janosh Riebesell, Rokas Elijosius"
__date__ = "2022-04-25"


# %%
epochs = 300
target_col = "e_form"
run_name = f"roost-robust-{epochs=}-{target_col}"
folds = list(range(5))
today = f"{datetime.now():%Y-%m-%d}"

if "roost" in run_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [k for k, v in mbv01_metadata.items() if v.input_type == "structure"]

log_dir = f"{os.path.dirname(__file__)}/job-logs/{today}-{run_name}"

slurm_submit_python(
    job_name=run_name,
    partition="ampere",
    account="LEE-SL3-GPU",
    time="8:0:0",
    array=f"0-{len(datasets) * len(folds) - 1}",
    log_dir=log_dir,
    slurm_flags=("--nodes 1", "--gpus-per-node 1"),
    # prepend into sbatch script to source module command and load default env
    # for Ampere GPU partition before actual job command
    pre_cmd=". /etc/profile.d/modules.sh; module load rhel8/default-amp;",
)


# %%
n_attn_layers = 3
embedding_aggregations = ("mean",)
optimizer = "AdamW"
learning_rate = 3e-4
df_or_path = f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz"
checkpoint = "wandb"  # None | 'local' | 'wandb'
batch_size = 128
swa_start = None
slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

print(f"Job started running {datetime.now():%Y-%m-%d@%H-%M}")
print(f"{run_name=}")
print(f"{df_or_path=}")

dataset_name, fold = list(product(datasets, folds))[slurm_array_task_id]
print(f"{dataset_name=}\n{fold=}")

train_wrenformer_on_matbench(
    model_name=run_name,
    dataset_name=dataset_name,
    timestamp=f"{datetime.now():%Y-%m-%d@%H-%M-%S}",
    fold=fold,
    epochs=epochs,
    n_attn_layers=n_attn_layers,
    checkpoint=checkpoint,
    learning_rate=learning_rate,
    embedding_aggregations=embedding_aggregations,
    embedding_type="composition" if "roost" in run_name.lower() else "wyckoff",
)
