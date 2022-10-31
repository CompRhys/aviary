# %%
import os
from datetime import datetime

from matbench_discovery.slurm import slurm_submit_python

from aviary import ROOT
from aviary.wrenformer.train import train_wrenformer_on_df

"""
Train a Wrenformer
ensemble of size n_folds on target_col of df_or_path.
"""

__author__ = "Janosh Riebesell"
__date__ = "2022-06-13"


# %%
epochs = 300
target_col = "e_form"
run_name = f"wrenformer-robust-{epochs=}-{target_col}"
n_folds = 10
today = f"{datetime.now():%Y-%m-%d}"
dataset = "mp"
log_dir = f"{os.path.dirname(__file__)}/{dataset}/{today}-{run_name}"

slurm_submit_python(
    job_name=run_name,
    partition="ampere",
    account="LEE-SL3-GPU",
    time="8:0:0",
    array=f"1-{n_folds}",
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

slurm_job_id = os.environ.get("SLURM_JOB_ID")
slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

print(f"Job started running {datetime.now():%Y-%m-%d@%H-%M}")
print(f"{slurm_job_id=}")
print(f"{slurm_array_task_id=}")
print(f"{run_name=}")
print(f"{df_or_path=}")

train_wrenformer_on_df(
    run_name=run_name,
    target_col=target_col,
    df_or_path=df_or_path,
    timestamp=f"{datetime.now():%Y-%m-%d@%H-%M-%S}",
    test_size=0.05,
    # folds=(n_folds, slurm_array_task_id),
    epochs=epochs,
    n_attn_layers=n_attn_layers,
    checkpoint=checkpoint,
    optimizer=optimizer,
    learning_rate=learning_rate,
    embedding_aggregations=embedding_aggregations,
    batch_size=batch_size,
    swa_start=swa_start,
    wandb_path="aviary/mp-wbm",
)
