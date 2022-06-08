# %%
import wandb
from tqdm import tqdm

__author__ = "Janosh Riebesell"
__date__ = "2022-05-18"

"""
Update run metadata recorded on [Weights and Biases](https://wandb.ai/aviary/matbench).
"""


# %%
wandb.login()
wandb_api = wandb.Api()


# %%
search_key = "display_name"
old_str = "n_transformer_layers"
new_str = "n_attn_layers"

runs = wandb_api.runs(
    "aviary/matbench", filters={search_key: {"$regex": f".*{old_str}.*"}}
)

print(f"matching runs: {len(runs)}")


# %% --- Update run metadata ---
for run in tqdm(runs):
    run.config[search_key] = run.config[search_key].replace(old_str, new_str)
    run.config["model"] = run.config["model"].replace(old_str, new_str)
    if old_str in run.config:
        run.config[new_str] = run.config.pop(old_str)
    run.update()
