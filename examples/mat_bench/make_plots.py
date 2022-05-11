# %%
import json
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from matbench import MatbenchBenchmark
from matbench.constants import CLF_KEY, REG_KEY
from matbench.task import MatbenchTask

from examples.mat_bench import DATA_PATHS
from examples.mat_bench.plotting_functions import (
    error_heatmap,
    plot_leaderboard,
    scale_errors,
    x_labels,
)
from examples.mat_bench.utils import open_json

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"

timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"


# %%
mb_repo = "/Users/janosh/dev/matbench"  # path to clone of matbench repo
others_data: dict[str, dict[str, float]] = defaultdict(dict)

# load benchmark data for models with existing Matbench submission
for dir_name in glob(f"{mb_repo}/benchmarks/*"):
    model_name = dir_name.split("/")[-1]

    mbbm = MatbenchBenchmark.from_file(f"{dir_name}/results.json.gz")

    for task in mbbm.tasks:
        task_name = task.dataset_name
        task_type = task.metadata.task_type

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        others_data[task_name][model_name] = score

df_err = pd.DataFrame(others_data).round(3)
df_err.index.name = "model"


# %%
for filename in glob("benchmarks/scores-*.json"):

    with open(filename) as file:
        data = json.load(file)["scores"]

    model_name = filename.split("benchmarks/scores-")[-1].split(".json")[0]
    df_mean_scores = pd.DataFrame(
        {task: pd.DataFrame(data[task]).mean(1) for task in data}
    ).T

    score_name = "mae" if "mae" in df_mean_scores else "rocauc"
    df_err.loc[model_name] = df_mean_scores[score_name]


# %%
our_benchmarks = glob("benchmarks/*.json.gz")

tasks = {}

for benchmark_path in our_benchmarks:
    bench_name = benchmark_path.split("/")[-1].split(".")[0]
    if bench_name in df_err.index:
        print(f"skipping {bench_name} since already in df_err.index")
        continue
    with open_json(benchmark_path) as json_data:
        benchmark = json_data

    for dataset, folds in benchmark.items():
        if sorted(folds) != list(range(5)):
            print(f"skipping partially recorded {dataset} with folds {sorted(folds)}")
            continue

        if dataset not in tasks:
            # load and cache task data manually without hydrating pymatgen structures for speed
            task = MatbenchTask(dataset, autoload=False)
            tasks[dataset] = task
            df = pd.read_json(DATA_PATHS[dataset]).set_index("mbid", drop=False)
            task.df = df
        else:
            task = tasks[dataset]

        task_type = task.metadata.task_type

        for fold, preds in folds.items():
            if task_type == CLF_KEY:
                preds = np.argmax(preds, axis=1)

            task.record(fold, preds)

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        others_data[dataset][bench_name] = score


# %%
df_err_scaled = scale_errors(df_err)
# rename column names for prettier axis ticks (must be after scale_errors() to have
# correct dict keys)
df_err_scaled = df_err_scaled.rename(columns=x_labels)


# %%
fig = plot_leaderboard(df_err)
fig.show()


# %%
html_path = f"plots/matbench-scaled-errors-{timestamp}.html"
# fig.write_html(html_path, include_plotlyjs="cdn")

# change plot background to black since Matbench site uses dark mode
with open(html_path, "r+") as file:
    html = file.read()
    file.seek(0)  # rewind file pointer to start of file
    html = html.replace(
        "</head>", "<style>body { background-color: black; }</style></head>"
    )
    file.write(html)
    file.truncate()


# %%
fig = error_heatmap(df_err_scaled)
fig.show()
fig.write_image(f"plots/matbench-scaled-errors-heatmap-{timestamp}.png", scale=2)
