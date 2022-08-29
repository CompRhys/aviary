from __future__ import annotations

from datetime import datetime

import pandas as pd
import wandb

from aviary import ROOT
from examples.wrenformer.deploy_wrenformer import deploy_wandb_checkpoints

__author__ = "Janosh Riebesell"
__date__ = "2022-08-15"

"""
Script that downloads checkpoints for an ensemble of Wrenformer models trained on the MP
formation energies, then makes predictions on some dataset, prints ensemble metrics and
stores predictions to CSV.
"""

today = f"{datetime.now():%Y-%m-%d}"

data_path = (  # download wbm-steps-summary.csv (23.31 MB)
    "https://figshare.com/ndownloader/files/36714216?private_link=ff0ad14505f9624f0c05"
)
df = pd.read_csv(data_path).set_index("material_id")


target_col = "e_form_per_atom"
df[target_col] = df.e_form / df.n_sites

wandb.login()
wandb_api = wandb.Api()
runs = wandb_api.runs(
    "aviary/mp", filters={"tags": {"$in": ["wrenformer-e_form-ensemble-1"]}}
)

df, ensemble_metrics = deploy_wandb_checkpoints(
    runs, df, input_col="wyckoff", target_col=target_col
)

df.to_csv(f"{ROOT}/examples/wrenformer/mp/{today}-ensemble-predictions.csv")
