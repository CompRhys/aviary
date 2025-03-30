# Matbench

This directory contains the files needed to create Matbench submissions for Roostformer and Wrenformer (structure tasks only for Wren) which are rewrites of Roost and Wren using PyTorch's builtin `TransformerEncoder` instead of the custom self-attention modules used by Roost and Wren.

Added in [aviary#44](https://github.com/CompRhys/aviary/pull/44).

## Speed difference between Wren and Wrenformer

According to Rhys, Wren could run 500 epochs in 5.5 h on a P100 training on 120k samples of MP data (similar to the `matbench_mp_e_form` dataset with 132k samples). Wrenformer only managed 207 epochs in 4h on the more powerful A100 training on `matbench_mp_e_form`. However, to avoid out-of-memory issues, Rhys constrained Wren to only run on systems with <= 16 Wyckoff positions. The code below shows that this lightens the workload by a factor of about 7.5, likely explaining the apparent slowdown in Wrenformer.

```py
import os
import sys
from pathlib import Path

# Add the parent directory to system path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from pymatgen.analysis.prototypes import count_wyckoff_positions
from matbench_example import DATA_PATHS

df = pd.read_json(DATA_PATHS["matbench_mp_e_form"])

df["n_wyckoff"] = df.wyckoff.map(count_wyckoff_positions)


sum_wyckoffs_sqr = (df.n_wyckoff**2).sum()
sum_wyckoffs_lte_16_sqr = (df.query("n_wyckoff <= 16").n_wyckoff ** 2).sum()
print(f"{sum_wyckoffs_sqr=}")
print(f"{sum_wyckoffs_lte_16_sqr=}")
print(f"{sum_wyckoffs_sqr/sum_wyckoffs_lte_16_sqr=:.3}")
# prints 7.45, so Wrenformer has to do 7.45x more work, explaining the about 2x slow down
# on a more powerful GPU (Nvidia A100 vs Wren on a P100)
```

## Benchmarks

JSON files in `model_scores/` contain only the calculated scores (MAE/ROCAUC) for a given model run. Files with the same name in `model_preds/` contain the full set of model predictions, targets and material ids. Code for loading these into memory is in `make_plots.py`.
