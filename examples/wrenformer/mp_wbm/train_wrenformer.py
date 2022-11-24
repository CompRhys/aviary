# %%
import os
from datetime import datetime

import pandas as pd
from matbench_discovery.slurm import slurm_submit

from aviary import ROOT
from aviary.train import df_train_test_split, train_wrenformer

"""
Train a Wrenformer ensemble of size n_folds on target_col of data_path.
"""

__author__ = "Janosh Riebesell"
__date__ = "2022-06-13"


# %%
epochs = 30
target_col = "e_form"
input_col = "wyckoff"
run_name = f"wrenformer-robust-mp+wbm-{epochs=}-{target_col}"
n_folds = 10
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M-%S}"
today = timestamp.split("@")[0]
log_dir = f"{os.path.dirname(__file__)}/{today}-{run_name}"

slurm_submit(
    job_name=run_name,
    partition="ampere",
    account="LEE-SL3-GPU",
    time="1:0:0",
    array=f"1-{n_folds}",
    log_dir=log_dir,
    slurm_flags=("--nodes", "1", "--gpus-per-node", "1"),
    # prepend into sbatch script to source module command and load default env
    # for Ampere GPU partition before actual job command
    pre_cmd=". /etc/profile.d/modules.sh; module load rhel8/default-amp;",
)


# %%
learning_rate = 3e-4
# data_path = f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz"
# for faster testing/debugging
data_path = f"{ROOT}/datasets/2022-06-09-mp+wbm-1k-samples.json.gz"
batch_size = 128
slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

print(f"Job started running {timestamp}")
print(f"{run_name=}")
print(f"{data_path=}")

df = pd.read_json(data_path).set_index("material_id", drop=False)
assert target_col in df, f"{target_col=} not in {list(df)}"
assert input_col in df, f"{input_col=} not in {list(df)}"
train_df, test_df = df_train_test_split(df, test_size=0.3)

run_params = dict(
    batch_size=batch_size,
    train_df=dict(shape=str(train_df.shape), columns=", ".join(train_df)),
    test_df=dict(shape=str(test_df.shape), columns=", ".join(test_df)),
)

train_wrenformer(
    run_name=run_name,
    train_df=train_df,
    test_df=test_df,
    target_col=target_col,
    task_type="regression",
    timestamp=timestamp,
    epochs=epochs,
    checkpoint="wandb",  # None | 'local' | 'wandb',
    learning_rate=learning_rate,
    batch_size=batch_size,
    wandb_path="aviary/mp-wbm",
    run_params=run_params,
)
