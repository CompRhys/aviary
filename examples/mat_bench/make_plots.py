# %%
import gzip
import json
from collections import defaultdict
from glob import glob

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

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"


# %%
all_data: dict[str, dict[str, float]] = defaultdict(dict)

mb_repo = "/Users/janosh/dev/matbench"  # path to clone of matbench repo

# load benchmark data for other models
for dir_name in glob(f"{mb_repo}/benchmarks/*"):
    model_name = dir_name.split("/")[-1]
    print(f"\n{model_name}")

    # results are automatically validated, no need to validate again
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

        all_data[task_name][model_name] = score


# %%
# df = pd.DataFrame(all_data).round(3)
# df.index.name = "model"
# df.to_csv("matbench-model-errors.csv")
df = pd.read_csv("matbench-model-errors.csv").set_index("model")
df_scaled = scale_errors(df)
# rename column names for prettier axis ticks (must be after scale_errors() to have
# correct dict keys)
df_scaled = df_scaled.rename(columns=x_labels)


# %%
def int_keys(d):
    # convert digit keys to ints in JSON dicts
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()}


our_benchmarks = [x for x in glob("benchmarks/*.json.gz") if x not in df.index]
for benchmark_path in our_benchmarks:
    bench_name = benchmark_path.split("/")[-1].split(".")[0]
    with gzip.open(benchmark_path) as json_gz:
        benchmark = json.load(json_gz, object_hook=int_keys)
    for dataset, folds in benchmark.items():
        if sorted(folds) != list(range(5)):
            print(f"skipping partially recorded {dataset} with folds {sorted(folds)}")
            continue
        task = MatbenchTask(dataset)
        df = pd.read_json(DATA_PATHS[dataset]).set_index("mbid", drop=False)
        task.df = df
        task_type = task.metadata.task_type
        for fold, preds in folds.items():
            task.record(fold, preds)

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        all_data[dataset][bench_name] = score


# %%
fig = plot_leaderboard(df)
fig.show()


# %%
html_path = "plots/matbench-scaled-errors.html"
fig.write_html(html_path, include_plotlyjs="cdn")


with open(html_path, "r+") as file:
    html = file.read()
    file.seek(0)  # rewind file pointer to start of file
    html = html.replace(
        "</head>", "<style>body { background-color: black; }</style></head>"
    )
    file.write(html)
    file.truncate()


# %%
fig = error_heatmap(df_scaled)
fig.show()
fig.write_image("plots/matbench-scaled-errors-heatmap.png")
