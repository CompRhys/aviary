# %%
import os
from datetime import datetime
from itertools import product

import pandas as pd
import wandb
from matbench.metadata import mbv01_metadata
from matbench.task import MatbenchTask
from matbench_discovery.slurm import slurm_submit_python

from aviary.core import TaskType
from aviary.train import train_wrenformer
from examples.wrenformer.mat_bench import DATA_PATHS
from examples.wrenformer.mat_bench.utils import merge_json_on_disk

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"

MODULE_DIR = os.path.dirname(__file__)

"""
Train a Wrenformer ensemble of size n_folds on collection of Matbench datasets.
"""


# %%
epochs = 300
folds = list(range(5))
today = f"{datetime.now():%Y-%m-%d}"
# job_name unlike run_name doesn't include dataset and fold since not yet known without
# SLURM_ARRAY_TASK_ID
job_name = f"matbench-wrenformer-robust-{epochs=}"
log_dir = f"{os.path.dirname(__file__)}/{today}-{job_name}"

if "roost" in job_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [k for k, v in mbv01_metadata.items() if v.input_type == "structure"]

slurm_submit_python(
    job_name=job_name,
    partition="ampere",
    account="LEE-SL3-GPU",
    time="8:0:0",
    array=f"0-{len(datasets) * len(folds) - 1}",
    log_dir=log_dir,
    slurm_flags=("--nodes", "1", "--gpus-per-node", "1"),
    # prepend into sbatch script to source module command and load default env
    # for Ampere GPU partition before actual job command
    pre_cmd=". /etc/profile.d/modules.sh; module load rhel8/default-amp;",
)


# %%
slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 35))
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M-%S}"
print(f"Job started running {timestamp}")

dataset_name, fold = list(product(datasets, folds))[slurm_array_task_id]
print(f"{dataset_name=}")
print(f"{fold=}")


data_path = DATA_PATHS[dataset_name]
id_col = "mbid"
df = pd.read_json(data_path).set_index(id_col, drop=False)

matbench_task = MatbenchTask(dataset_name, autoload=False)
matbench_task.df = df

target_col = matbench_task.metadata.target
run_name = f"{job_name}-{dataset_name}-{fold=}-{target_col}"
print(f"{run_name=}")
task_type: TaskType = matbench_task.metadata.task_type

train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

test_metrics, run_params, test_df = train_wrenformer(
    checkpoint="wandb",  # None | 'local' | 'wandb'
    run_name=run_name,
    train_df=train_df,
    test_df=test_df,
    target_col=target_col,
    task_type=task_type,
    id_col=id_col,
    # set to None to disable logging
    wandb_path="aviary/matbench",
    run_params=dict(dataset=dataset_name, fold=fold),
    timestamp=timestamp,
    epochs=epochs,
)

# save model predictions to JSON
preds_path = f"{MODULE_DIR}/model_preds/{timestamp}-{run_name}.json"

# record model predictions
preds_dict = test_df[[id_col, target_col, f"{target_col}_pred"]].to_dict(orient="list")
merge_json_on_disk({dataset_name: {f"fold_{fold}": preds_dict}}, preds_path)

# save model scores to JSON
scores_path = f"{MODULE_DIR}/model_scores/{timestamp}-{run_name}.json"
scores_dict = {dataset_name: {f"fold_{fold}": test_metrics}}
scores_dict["params"] = run_params
scores_dict["wandb_run"] = wandb.run.get_url()
merge_json_on_disk(scores_dict, scores_path)

print(f"scores for {fold = } of task {dataset_name} written to {scores_path}")
