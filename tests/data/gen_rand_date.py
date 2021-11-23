# writes the files tests/data/rand_{regr,clf}.csv
import numpy as np
import pandas as pd

xs = np.random.rand(100)
y_pred = xs + 0.1 * np.random.normal(size=100)
y_true = xs + 0.1 * np.random.normal(size=100)

df_regr = pd.DataFrame({"xs": xs, "y_pred": y_pred, "y_true": y_true})
df_regr.to_csv("tests/data/rand_regr.csv", index=False, float_format="%g")

y_binary = np.random.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np.random.normal(scale=5, size=100), 0.2, 0.9)

df_clf = pd.DataFrame(
    {"y_binary": y_binary, "y_proba": y_proba, "y_pred": y_proba.round()}
)
df_clf.to_csv("tests/data/rand_clf.csv", index=False, float_format="%g")
