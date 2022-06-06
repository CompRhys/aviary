# %%
import json
import logging
from collections import defaultdict
from datetime import datetime
from glob import glob

import pandas as pd
import plotly.express as px
from matbench import MatbenchBenchmark
from matbench.constants import CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata as matbench_metadata
from pymatviz.utils import add_identity_line
from sklearn.metrics import r2_score, roc_auc_score

from examples.mat_bench import DATA_PATHS
from examples.mat_bench.plotting_functions import (
    dataset_labels_html,
    error_heatmap,
    plot_leaderboard,
    scale_errors,
)
from examples.mat_bench.utils import dict_merge

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"

logging.getLogger("matbench").setLevel("ERROR")

timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"
matbench_repo_path = "/Users/janosh/dev/matbench"  # path to clone of matbench repo
bench_dir = f"{matbench_repo_path}/benchmarks"

our_scores: dict[str, dict[str, float]] = defaultdict(dict)
others_scores: dict[str, dict[str, float]] = defaultdict(dict)


# %% --- Load other's scores ---
# load benchmark data for models with existing Matbench submission
for idx, dirname in enumerate(glob(f"{bench_dir}/*"), 1):
    model_name = dirname.split("/matbench_v0.1_")[-1]
    print(f"{idx}. {model_name}")
    mbbm = MatbenchBenchmark.from_file(f"{dirname}/results.json.gz")

    for task in mbbm.tasks:
        task_name = task.dataset_name
        task_type = task.metadata.task_type

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        others_scores[model_name][task_name] = round(score, 3)


# %% --- Load our scores ---
our_score_files = sorted(
    glob("model_scores/*.json"), key=lambda s: s.split("-2022-")[-1]
)

for idx, filename in enumerate(our_score_files, 1):
    model_name, date = filename.split("/")[-1].split("-2022-")
    date = date.split("@")[0]
    print(f"{idx}. {date=} {model_name}")

    with open(filename) as file:
        data = json.load(file)

    # filter params and other unwanted keys
    data = {k: data[k] for k in data if k.startswith("matbench_")}

    mean_scores = {}
    for task in data:
        df_fold_means = pd.DataFrame(data[task]).mean(1).round(3)
        key = "mae" if "mae" in df_fold_means else "rocauc"
        mean_scores[task] = df_fold_means[key]

    # mean_scores["date"] = f"2022-{date}"
    our_scores[model_name] = mean_scores


# %%
matbench_dfs: dict[str, pd.DataFrame] = {}
tasks_to_load = ["matbench_mp_e_form"]
# tasks_to_load = list(DATA_PATHS)

for task_name in tasks_to_load:
    if task_name in matbench_dfs:
        continue
    task_df = pd.read_json(DATA_PATHS[task_name]).set_index("mbid")
    task_df = task_df.drop(columns=["structure"], errors="ignore")
    matbench_dfs[task_name] = task_df


# %% --- Load other's predictions ---
for task_name in tasks_to_load:
    for model_name in ["alignn", "Crabnet"]:
        results_path = f"{bench_dir}/matbench_v0.1_{model_name}/results.json.gz"

        with open(results_path) as file:
            data = json.load(file)["tasks"][task_name]["results"]

            join_folds: dict[str, float] = {}
            for idx in range(5):
                join_folds |= data[f"fold_{idx}"]["data"]

            task_df = matbench_dfs[task_name]

            task_df[model_name] = pd.Series(join_folds)


# %% --- Load our predictions ---
our_pred_files = sorted(glob("model_preds/*swa*.json"))
pred_col = "predictions"

for file_path in our_pred_files:
    model_name = file_path.split("/")[-1].split("-2022")[0]
    print(f"\nReading {model_name}...")
    with open(file_path) as file:
        json_data = json.load(file)

    # for task_name, folds in json_data.items():  # loads all tasks
    for task_name in tasks_to_load:  # loads only selected tasks
        folds = json_data[task_name]
        if len(folds) != 5:
            print(f"  {task_name} only partially recorded: {sorted(folds)}")

        target = matbench_metadata[task_name].target
        task_type = matbench_metadata[task_name].task_type

        dfs = {idx: pd.DataFrame(fold_dict) for idx, fold_dict in folds.items()}

        task_df = pd.concat(dfs.values())
        task_df = task_df.set_index("mbid")

        if task_type == CLF_KEY:
            proba_cls_1 = task_df[pred_col].str[1]
            folds_mean_score = roc_auc_score(task_df[target], proba_cls_1)
        else:
            folds_mean_score = (task_df[target] - task_df[pred_col]).abs().mean()
        our_scores[model_name][task_name] = round(folds_mean_score, 4)

        # record our model preds into new df columns
        matbench_dfs[task_name][model_name] = task_df[pred_col]


# %%
df_err = pd.DataFrame(dict_merge(our_scores, others_scores))
df_err.index.name = "dataset"
print(f"{df_err.shape=}")
df_err


# %%
# html_path = f"plots/matbench-leaderboard-{timestamp}.html"
html_path = None
plot_leaderboard(df_err.dropna(axis=1, thresh=5), html_path, width=1200, height=600)


# %% error heatmap using Pandas dataframe styler
# thresh=x means require at least x non-NA values
df_err_scaled = (
    scale_errors(df_err).dropna(thresh=4).dropna(thresh=8, axis=1)
)  # .drop("matbench_jdft2d")

df_err_scaled.loc["mean scaled error"] = df_err_scaled.mean(0)

df_display = df_err_scaled.T.sort_values(by="mean scaled error")
df_display = df_display.rename(columns=dataset_labels_html)

df_display.style.format(precision=3).background_gradient(cmap="viridis")


# %% error heatmap using plotly
# thresh=x means require at least x non-NA values
fig = error_heatmap(df_err.dropna(thresh=9, axis=1), width=1200, height=600)
fig.show()
fig.write_image(f"plots/matbench-scaled-errors-heatmap-{timestamp}.png", scale=2)


# %% scatter plot of predictions vs. targets for multiple models on matbench_mp_e_form
df = matbench_dfs["matbench_mp_e_form"]
df = df.dropna(axis=1)  # drop models with missing predictions
target = df.columns[0]


y_cols = [c for c in df if c not in [target, "composition", "wyckoff"]]
labels = {}

for y_col in y_cols:
    MAE = (df[y_col] - df[target]).abs().mean()
    pretty_title = y_col.replace("-", " ")
    R2 = r2_score(df[target], df[y_col])
    labels[y_col] = f"{pretty_title}<br>{MAE = :.2f}, {R2 = :.2f}"

fig = px.scatter(
    df.rename(columns=labels).reset_index(),
    x=target,
    y=list(labels.values()),
    hover_data=["mbid"],
    opacity=0.7,
    width=1200,
    height=800,
    labels={
        "e_form": "DFT formation energy (eV/atom)",
        "value ": "Predicted formation energy (eV/atom)",
    },
)
add_identity_line(fig)

fig.update_layout(legend=dict(x=0.02, y=0.95, xanchor="left", title="Models"))

# fig.write_image(f"plots/matbench-mp-e-form-scatter-{timestamp}.png", scale=2)
