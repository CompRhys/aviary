# Matbench

Work in progress preparing Matbench submissions for Roost and Wren (structure tasks only for Wren).

Directory is named `mat_bench` to avoid shadowing the `matbench` package.

## Speed difference between Wren and Wrenformer

According to Rhys, Wren could run 500 epochs in 5.5 h on a P100 training on 120k samples of MP data (similar to the `matbench_mp_e_form` dataset with 132k samples). Wrenformer only managed 207 epochs in 4h on the more powerful A100 training on `matbench_mp_e_form`. However, to avoid out-of-memory issues, Rhys constrained Wren to only run on systems with <= 16 Wyckoff positions. The code below shows that this lightens the workload by a factor of about 7.5, likely explaining the apparent slowdown in Wrenformer.

```py
import pandas as pd
from aviary.wren.utils import count_wyks
from examples.mat_bench import DATA_PATHS

df = pd.read_json(DATA_PATHS["matbench_mp_e_form"])

df["n_wyckoff"] = df.wyckoff.map(count_wyks)


sum_wyckoffs_sqr = (df.n_wyckoff**2).sum()
sum_wyckoffs_lte_16_sqr = (df.query("n_wyckoff <= 16").n_wyckoff ** 2).sum()
print(f"{sum_wyckoffs_sqr=}")
print(f"{sum_wyckoffs_lte_16_sqr=}")
print(f"{sum_wyckoffs_sqr/sum_wyckoffs_lte_16_sqr=:.3}")
# prints 7.45, so Wrenformer has to do 7.45x more work, explaining the about 2x slow down
# on a more powerful GPU (Nvidia A100 vs Wren on a P100)
```
