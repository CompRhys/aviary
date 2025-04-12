"""Train a Wrenformer ensemble of size n_folds on collection of Matbench datasets."""

# %%
import os
import sys
from pathlib import Path

# Add the parent directory to system path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime
from itertools import product

import wandb
from matbench.metadata import mbv01_metadata
from matbench.task import MatbenchTask
from matbench_example.prepare_matbench_datasets import DATA_PATHS
from matbench_example.trainer import train_wrenformer
from matbench_example.utils import merge_json_on_disk, slurm_submit
from matminer.utils.io import load_dataframe_from_json

from aviary.core import TaskType

MODULE_DIR = os.path.dirname(__file__)


# %%
epochs = 10
folds = list(range(5))
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M-%S}"
today = timestamp.split("@")[0]
# job_name unlike run_name doesn't include dataset and fold since not yet known without
# SLURM_ARRAY_TASK_ID
job_name = f"matbench-wrenformer-robust-{epochs=}"
out_dir = f"{os.path.dirname(__file__)}/{today}-{job_name}"

if "roost" in job_name.lower():
    # deploy Roost on all tasks
    datasets = list(DATA_PATHS)
else:
    # deploy Wren on structure tasks only
    datasets = [
        k
        for k, v in mbv01_metadata.items()
        if v.input_type == "structure" and k in DATA_PATHS
    ]

# NOTE: this script will run as is if you want to run it locally without slurm.
slurm_submit(
    job_name=job_name,
    partition="ampere",
    account="LEE-SL3-GPU",
    time="8:0:0",
    array=f"0-{len(datasets) * len(folds) - 1}",
    out_dir=out_dir,
    slurm_flags=("--nodes", "1", "--gpus-per-node", "1"),
    # prepend into sbatch script to source module command and load default env
    # for Ampere GPU partition before actual job command
    pre_cmd=". /etc/profile.d/modules.sh; module load rhel8/default-amp;",
)


# %%
slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
print(f"Job started running {timestamp}")

dataset_name, fold = list(product(datasets, folds))[slurm_array_task_id]
print(f"{dataset_name=}")
print(f"{fold=}")


data_path = DATA_PATHS[dataset_name]
id_col = "mbid"
df = load_dataframe_from_json(data_path)
df.index.name = id_col

matbench_task = MatbenchTask(dataset_name, autoload=False)
matbench_task.df = df

target_col = matbench_task.metadata.target
run_name = f"{job_name}-{dataset_name}-{fold=}-{target_col}"
print(f"{run_name=}")
task_type: TaskType = matbench_task.metadata.task_type

train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

wandb_path = None

test_metrics, run_params, test_df = train_wrenformer(
    checkpoint=None,  # None | 'local' | 'wandb'
    run_name=run_name,
    train_df=train_df,
    test_df=test_df,
    target_col=target_col,
    task_type=task_type,
    id_col=id_col,
    # set to None to disable logging
    wandb_path=wandb_path,
    run_params=dict(dataset=dataset_name, fold=fold),
    timestamp=timestamp,
    epochs=epochs,
)


# %%
# save model predictions to JSON
preds_path = f"{MODULE_DIR}/model_preds/{timestamp}-{run_name}.json"
os.makedirs(os.path.dirname(preds_path), exist_ok=True)

# record model predictions
test_df[id_col] = test_df.index
preds_dict = test_df[[id_col, target_col, f"{target_col}_pred_0"]].to_dict(orient="list")
merge_json_on_disk({dataset_name: {f"fold_{fold}": preds_dict}}, preds_path)

# save model scores to JSON
scores_path = f"{MODULE_DIR}/model_scores/{timestamp}-{run_name}.json"
os.makedirs(os.path.dirname(scores_path), exist_ok=True)

scores_dict = {dataset_name: {f"fold_{fold}": test_metrics}}
scores_dict["params"] = run_params
if wandb_path is not None:
    scores_dict["wandb_run"] = wandb.run.get_url()
merge_json_on_disk(scores_dict, scores_path)

print(f"scores for {fold = } of task {dataset_name} written to {scores_path}")
