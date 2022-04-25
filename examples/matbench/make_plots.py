# %%
from collections import defaultdict
from glob import glob

import pandas as pd
from matbench import MatbenchBenchmark
from matbench.constants import CLF_KEY, REG_KEY
from monty.serialization import loadfn

from examples.matbench.plotting_functions import plot_scaled_errors

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"

# %%
all_data: dict[str, dict[str, float]] = defaultdict(dict)
BENCHMARKS_DIR = "/Users/janosh/dev/matbench/benchmarks"

# Get all benchmark data loaded into memory
for dir_name in glob(f"{BENCHMARKS_DIR}/*"):
    print(dir_name)

    # results are automatically validated, no need to validate again
    mbbm = MatbenchBenchmark.from_file(f"{dir_name}/results.json.gz")

    info = loadfn(f"{dir_name}/info.json")

    algo = info["algorithm"]

    for task in mbbm.tasks:
        task_name = task.dataset_name
        task_type = task.metadata.task_type

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        all_data[task_name][algo] = score


# %%
for bench_name in ["wren-4of9-2022-04-15@13-31", "roost-2022-04-15@19-21"]:
    benchmark = MatbenchBenchmark.from_file(f"benchmarks/{bench_name}.json")
    for task in benchmark.tasks:
        task_name = task.dataset_name
        task_type = task.metadata.task_type

        if task_type == REG_KEY:
            score = task.scores.mae.mean
        elif task_type == CLF_KEY:
            score = task.scores.rocauc.mean
        else:
            raise ValueError(f"Unknown {task_type = }")

        all_data[task_name][bench_name] = score


# %%
df = pd.DataFrame(all_data)
fig, scaled_df = plot_scaled_errors(df)
fig.show()
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
