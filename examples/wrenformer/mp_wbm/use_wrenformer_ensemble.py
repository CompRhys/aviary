from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import wandb

from aviary import ROOT
from aviary.deploy import predict_from_wandb_checkpoints

__author__ = "Janosh Riebesell"
__date__ = "2022-06-23"

"""
Script that downloads checkpoints for an ensemble of Wrenformer models trained on
the MP+WBM formation energies and makes predictions on the test set, then prints
ensemble metrics and stores predictions to CSV.
"""

module_dir = os.path.dirname(__file__)
today = f"{datetime.now():%Y-%m-%d}"
data_path = f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz"
test_size = 0.05
df = pd.read_json(data_path)
# shuffle with same random seed as in train_wrenformer() to get identical train/test split
df = df.sample(frac=1, random_state=0)
train_df = df.sample(frac=1 - test_size, random_state=0)  # unused
test_df = df.drop(train_df.index)
target_col = "e_form"

wandb.login()
wandb_api = wandb.Api()
ensemble_id = "ensemble-id-2"
runs = wandb_api.runs("aviary/mp-wbm", filters={"tags": {"$in": [ensemble_id]}})

assert len(runs) == 10, f"Expected 10 runs, got {len(runs)} for {ensemble_id=}"

test_df, ensemble_metrics = predict_from_wandb_checkpoints(
    runs, test_df, input_col="wyckoff", target_col=target_col
)

test_df.round(6).to_csv(f"{module_dir}/{today}-{ensemble_id}-preds-{target_col}.csv")

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
