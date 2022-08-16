from __future__ import annotations

import os
from glob import glob

import pandas as pd
import wandb

from aviary import ROOT
from aviary.wrenformer.utils import make_ensemble_predictions

__author__ = "Janosh Riebesell"
__date__ = "2022-06-23"

"""
Script that downloads checkpoints for an ensemble of Wrenformer models trained on
the MP+WBM dataset and makes predictions on the test set, then prints ensemble metrics.
"""


data_path = f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz"
target_col = "e_form"
test_size = 0.05
df = pd.read_json(data_path)
# shuffle with same random seed as in train_wrenformer() to get identical train/test split
df = df.sample(frac=1, random_state=0)
train_df = df.sample(frac=1 - test_size, random_state=0)
test_df = df.drop(train_df.index)


load_checkpoints_from_wandb = True

if load_checkpoints_from_wandb:
    wandb.login()
    wandb_api = wandb.Api()

    runs = wandb_api.runs("aviary/mp-wbm", filters={"tags": {"$in": ["ensemble-id-2"]}})

    print(
        f"Loading checkpoints for the following run IDs:\n{', '.join(run.id for run in runs)}\n"
    )

    checkpoint_paths: list[str] = []
    for run in runs:
        run_path = "/".join(run.path)
        checkpoint_dir = f"{ROOT}/.wandb_checkpoints/{run_path}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{checkpoint_dir}/checkpoint.pth"
        checkpoint_paths.append(checkpoint_path)

        # download checkpoint from wandb if not already present
        if os.path.isfile(checkpoint_path):
            continue
        wandb.restore("checkpoint.pth", root=checkpoint_dir, run_path=run_path)
else:
    # load checkpoints from local run dirs
    checkpoint_paths = glob(
        f"{ROOT}/examples/mp_wbm/job-logs/wandb/run-20220621_13*/files/checkpoint.pth"
    )

print(f"Predicting with {len(checkpoint_paths):,} model checkpoint(s)")

test_df, ensemble_metrics = make_ensemble_predictions(
    checkpoint_paths, df=test_df, target_col=target_col
)

test_df.to_csv(f"{ROOT}/examples/mp_wbm/ensemble-predictions.csv")


# print output:
# Predicting with 10 model checkpoint(s)
#
# Single model performance:
#          MAE    RMSE      R2
# mean  0.0369  0.1218  0.9864
# std   0.0005  0.0014  0.0003
#
# Ensemble performance:
# MAE      0.0308
# RMSE     0.118
# R2       0.987
