import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition

plt.rcParams.update({"font.size": 20})

plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.major.width"] = 2.5
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 2.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.major.width"] = 2.5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["ytick.minor.width"] = 2.5
plt.rcParams["legend.fontsize"] = 20

fig, ax = plt.subplots(1, 1, figsize=(10, 9))

df_list = []
for i in range(0, 6):
    j, k = divmod(i, 3)
    offsets = 1

    if i < 5:
        title = r"$\bf{{{}}}$".format(f"Batch-{i+offsets}")
        df = pd.read_csv(
            f"results/manuscript/step_{i+offsets}_wren_org.csv",
            comment="#",
            na_filter=False,
        )

        print(len(df))
        df["is_el"] = df.composition.apply(lambda x: Composition(x).is_element)
        df = df[~df.is_el]
        df = df.drop(["is_el"], axis=1)
        print(len(df))

        df_list.append(df)

        df_hull = pd.read_csv(
            f"datasets/wbm-ehull/step-{i+offsets}-e_hull.csv",
            comment="#",
            na_filter=False,
        )
        mapping = dict(df_hull[["material_id", "E_hull"]].values)
        df["E_hull"] = pd.to_numeric(df["id"].map(mapping))
        df = df.dropna(axis=0, subset=["E_hull"])
        tar = df["E_hull"].to_numpy().ravel()

        ylim = (0, 3000)
        yticks = (0, 1000, 2000, 3000)

        continue
    else:
        title = r"$\bf{Combined}$"
        df = pd.concat(df_list)

        ylim = (0, 3000)
        yticks = (0, 1000, 2000, 3000)

    init = len(df)

    # rare = "all"

    rare = "nla"
    df = df[
        ~df["composition"].apply(
            lambda x: any(el.is_rare_earth_metal for el in Composition(x).elements)
        )
    ]

    # print(1-len(df)/init)

    tar = df["E_hull"].to_numpy().ravel()

    tar_cols = [col for col in df.columns if "target" in col]
    # tar = df[tar_cols].to_numpy().ravel() - e_hull
    tar_f = df[tar_cols].to_numpy().ravel()

    pred_cols = [col for col in df.columns if "pred" in col]
    pred = df[pred_cols].to_numpy().T
    # mean = np.average(pred, axis=0) - e_hull
    mean = np.average(pred, axis=0) - tar_f + tar

    epi = np.var(pred, axis=0, ddof=0)

    ale_cols = [col for col in df.columns if "ale" in col]
    ales = df[ale_cols].to_numpy().T
    ale = np.mean(np.square(ales), axis=0)

    both = np.sqrt(epi + ale)

    crit = "std"
    test = mean + both

    crit = "ene"
    test = mean

    bins = 200
    # xlim = (-0.2, 0.2)
    xlim = (-0.4, 0.4)
    # xlim = (-1, 1)

    alpha = 0.5
    # thresh = 0.02
    thresh = 0.00
    xticks = (-0.4, -0.2, 0, 0.2, 0.4)
    # yticks = (0, 300, 600, 900, 1200)

    e_type = "true"
    tp = tar[(tar <= thresh) & (test <= thresh)]
    fn = tar[(tar <= thresh) & (test > thresh)]
    fp = tar[(tar > thresh) & (test <= thresh)]
    tn = tar[(tar > thresh) & (test > thresh)]
    xlabel = (
        r"$\Delta$" + r"$\it{E}$" + r"$_{Hull-MP}$" + " / eV per atom"
    )  # r"$\/(\frac{eV}{atom})$"

    # n_test = None
    # sort = np.argsort(mean)
    # mean = mean[sort][:n_test]

    # e_type = "pred"
    # tp = mean[(tar[sort][:n_test] <= thresh) & (test[sort][:n_test] <= thresh)]
    # fn = mean[(tar[sort][:n_test] <= thresh) & (test[sort][:n_test] > thresh)]
    # fp = mean[(tar[sort][:n_test] > thresh) & (test[sort][:n_test] <= thresh)]
    # tn = mean[(tar[sort][:n_test] > thresh) & (test[sort][:n_test] > thresh)]
    # xlabel = (
    #     r"$\Delta$" + r"$\it{E}$" + r"$_{Hull-Pred}$" + " / eV per atom"
    # )  # r"$\/(\frac{eV}{atom})$"

    ax.hist(
        [tp, fn, fp, tn],
        bins=bins,
        range=xlim,
        alpha=alpha,
        color=["tab:green", "tab:orange", "tab:red", "tab:blue"],
        linestyle="none",
        label=[
            "True Positives",
            "False Negatives",
            "False Positives",
            "True Negatives",
        ],
        stacked=True,
    )

    ax.legend(frameon=False, loc="upper left")

    tp, fp, tn, fn = len(tp), len(fp), len(tn), len(fn)
    null = (tp + fn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp)
    tpr = tp / (tp + fn)
    f1 = 2 * ppv * tpr / (ppv + tpr)

    print(f"PPV: {ppv}")
    print(f"TPR: {tpr}")
    print(f"F1: {f1}")
    print(f"Enrich: {ppv/null}")

    xpos, ypos = 0.45 * xlim[1], 0.96 * ylim[1]
    fontsize = 20

    ax.text(
        xpos,
        ypos,
        # f"Prevalence = {null:.2f}\nPrecision = {ppv:.2f}\nRecall = {tpr:.2f}",
        f"Enrichment\nFactor = {ppv/null:.1f}",
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="left",
    )

    xpos, ypos = 0.90 * xlim[0], 0.96 * ylim[1]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xlabel(xlabel)

    ax.set_ylabel("Number of Compounds")
    # else:
    # ax.get_yaxis().set_ticklabels([])

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax.set_aspect(1.0 / ax.get_data_ratio())


fig.tight_layout()
# NOTE this figure plots hist bars separately which causes aliasing in pdf
# to resolve this take into inkscape and merge regions by colour
plt.savefig(f"examples/plots/pdf/hist-{e_type}-{crit}-{rare}.pdf")

plt.show()
