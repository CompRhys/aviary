# %%
from __future__ import annotations

import pandas as pd
import torch
from matbench.metadata import mbv01_metadata
from torch.nn import L1Loss
from tqdm import tqdm

from aviary.core import Normalizer
from aviary.data import InMemoryDataLoader
from aviary.wrenformer.data import collate_batch, get_initial_wyckoff_embedding
from aviary.wrenformer.model import Wrenformer
from examples.mat_bench import DATA_PATHS

__author__ = "Janosh Riebesell"
__date__ = "2022-04-12"

# %%
model_name = "wrenformer"

# deploy Wren on structure tasks only
structure_datasets = [
    k for k, v in mbv01_metadata.items() if v.input_type == "structure"
]


# %%
df = pd.read_json(DATA_PATHS["matbench_jdft2d"])
target = "exfoliation_en"
task_type = "regression"
task_dict = {target: task_type}


# %%
model = Wrenformer(
    robust=False,
    n_targets=[1],
    n_features=200 + 444 + 1,
    task_dict=task_dict,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Pytorch running on {device=}")

learning_rate = 3e-4
criterion_dict = {target: (task_type, L1Loss())}
normalizer_dict = {target: Normalizer()}


learning_rate = 1e-3
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
warmup_steps = 10


def lr_lambda(epoch: int) -> float:
    return min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=True)


# %%
df["features"] = [
    get_initial_wyckoff_embedding(wyk_str) for wyk_str in tqdm(df.wyckoff)
]

features, targets, ids = (df[x] for x in ["features", target, "mbid"])
targets = torch.tensor(targets, device=device)
features = tuple(tensor.to(device) for tensor in features)


mem_dataloader = InMemoryDataLoader(
    [features, targets, ids], batch_size=32, collate_fn=collate_batch
)


# %%
model.fit(
    mem_dataloader,
    mem_dataloader,
    optimizer,
    scheduler,
    epochs=10,
    criterion_dict=criterion_dict,
    normalizer_dict=normalizer_dict,
    model_name=model_name,
    run_id=1,
)
