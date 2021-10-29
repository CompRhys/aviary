# %%
# Import Libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from sklearn.metrics import r2_score

plt.rcParams.update({"font.size": 20})

plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.major.width"] = 2.5
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 2.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.major.width"] = 2.5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["ytick.minor.width"] = 2.5
plt.rcParams["legend.fontsize"] = 20

# %%


def get_spg(num):
    return int(num.split("_")[2])


# %%
# scatter plots

# TAATA
df_test = pd.read_csv("datasets/taata/taata-c-test.csv", comment="#", na_filter=False)
df_test = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
    df_test, "composition"
)

df_hull = pd.read_csv("datasets/taata/taata-c-train.csv", comment="#", na_filter=False)
df_hull = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
    df_hull, "composition"
)

entries = [
    PDEntry(c, e * c.num_atoms) for c, e in df_hull[["composition_obj", "E_f"]].values
]
el_entries = [PDEntry(c, 0) for c in ["N", "Zn", "Zr", "Ti", "Hf"]]

ppd = PhaseDiagram(entries + el_entries)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    e_hull_dict = {
        c: ppd.get_hull_energy(c) / c.num_atoms
        for c in df_test["composition_obj"].values
    }


# %%
# get colourmap
df_test["spacegroup"] = df_test["wyckoff"].apply(get_spg)
spg = df_test["spacegroup"].values
sort = np.argsort(spg)

fig, ax_scatter = plt.subplots(2, 2, figsize=(20, 15))

titles = ["Roost", r"Wren\ (This\ Work)", "CGCNN", "CGCNN"]
reps = [
    "Composition",
    r"Wyckoff\ Representation",
    "Pre \u2212 relaxation\\ Structures",
    r"Relaxed\ Structures",
]
fs = [
    "results/manuscript/multi_results_roost-taata-c_s-0_t-1.csv",
    "results/manuscript/multi_results_wren-taata-c_s-0_t-1.csv",
    "results/manuscript/multi_results_cgcnn-taata-un-c_s-0_t-1.csv",
    "results/manuscript/multi_results_cgcnn-taata-c_s-0_t-1.csv",
]

for i, (title, rep, f) in enumerate(zip(titles, reps, fs)):
    j, k = divmod(i, 2)
    df = pd.read_csv(f, comment="#", na_filter=False)
    e_hull = np.array([e_hull_dict[Composition(c)] for c in df["composition"]])

    tar_cols = [col for col in df.columns if "target" in col]
    tar = df[tar_cols].to_numpy().ravel() - e_hull

    pred_cols = [col for col in df.columns if "pred" in col]
    pred = df[pred_cols].to_numpy().T
    mean = np.average(pred, axis=0) - e_hull

    epi = np.var(pred, axis=0, ddof=0)

    ale_cols = [col for col in df.columns if "ale" in col]
    ales = df[ale_cols].to_numpy().T
    ale = np.mean(np.square(ales), axis=0)

    res = mean - tar
    mae = np.abs(res).mean()
    rmse = np.sqrt(np.square(res).mean())
    r2 = r2_score(tar, mean)

    ax_scatter[j, k].tick_params(direction="out")

    im = ax_scatter[j, k].scatter(
        tar[sort],
        mean[sort],
        c=spg[sort],
        cmap=plt.get_cmap("turbo_r", 230),
        s=16,
        alpha=0.8,
        rasterized=True,
    )  # vmax=230, vmin=1)

    if j == 1:
        ax_scatter[j, k].set_xlabel(
            r"$\it{E}$" + r"$_{Hull-TAATA}$" + " / eV per atom", labelpad=8
        )

    if k == 0:
        ax_scatter[j, k].set_ylabel(
            r"$\it{E}$" + r"$_{Hull-ML}$" + " / eV per atom", labelpad=6
        )

    # now determine nice limits by hand:
    binwidth = 0.05
    top = 3.5
    bottom = -0.5

    x_lims = np.array((bottom, top))
    y_lims = np.array((bottom, top))

    ax_scatter[j, k].plot(x_lims, y_lims, color="grey", linestyle="--", alpha=0.3)

    ax_scatter[j, k].set_xlim(x_lims)
    ax_scatter[j, k].set_ylim(y_lims)

    ax_scatter[j, k].set_xticks((0, 1, 2, 3))
    ax_scatter[j, k].set_yticks((0, 1, 2, 3))

    ax_scatter[j, k].annotate(
        f"$\\bf{{Input: {rep}}}$\n$\\bf{{Model: {title}}}$\n "
        f"$R^2$ = {r2:.2f}\n MAE = {mae:.2f}\n RMSE = {rmse:.2f}",
        (0.05, 0.72),
        xycoords="axes fraction",
    )

    ax_scatter[j, k].set_aspect(1.0 / ax_scatter[j, k].get_data_ratio())

plt.tight_layout()
plt.subplots_adjust(wspace=-0.4)
plt.savefig("examples/plots/pdf/taata-c-hull-all.pdf")

plt.show()
