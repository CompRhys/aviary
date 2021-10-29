# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition


def get_spg(num):
    return int(num.split("_")[2])


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
space = "mp"
df = pd.read_csv(f"results/manuscript/multi_results_chemsys-{space}_s-0_t-1.csv")

# %%
cols = [col for col in df.columns if "E_f" in col]
tar_col = [col for col in cols if "target" in col]
tar = df[tar_col].to_numpy().ravel()

pred_cols = [col for col in cols if "pred" in col]
pred = df[pred_cols].to_numpy().T
df["E_f"] = np.average(pred, axis=0)

epi = np.var(pred, axis=0, ddof=0)

ale_cols = [col for col in cols if "ale" in col]
ales = df[ale_cols].to_numpy().T
ale = np.mean(np.square(ales), axis=0)

df["std"] = np.sqrt(epi + ale)

df["comp_obj"] = df["composition"].apply(lambda x: Composition(x))
df["Hf"] = df["comp_obj"].apply(
    lambda x: frozenset(x.elements).issubset(Composition("HfZnN").elements)
)
df["Zr"] = df["comp_obj"].apply(
    lambda x: frozenset(x.elements).issubset(Composition("ZrZnN").elements)
)
df["Ti"] = df["comp_obj"].apply(
    lambda x: frozenset(x.elements).issubset(Composition("TiZnN").elements)
)
df["num_atoms"] = df["comp_obj"].apply(lambda x: x.num_atoms)


df = df.drop(pred_cols, axis=1)
df = df.drop(ale_cols, axis=1)

# %%

fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))

for k, el in enumerate(["Ti", "Zr", "Hf"]):

    Hf = df[df[el]]

    entries_ml = [
        PDEntry(c, e * n, idx)
        for c, e, n, idx in Hf[["comp_obj", "E_f", "num_atoms", "material_id"]].values
    ]
    entries_dft = [
        PDEntry(c, e * n, idx)
        for c, e, n, idx in Hf[
            ["comp_obj", "E_f_target", "num_atoms", "material_id"]
        ].values
    ]

    elems = [PDEntry(c.symbol, 0) for c in Composition(f"{el}ZnN").elements]

    pd_ml = PhaseDiagram(entries_ml + elems)
    pd_dft = PhaseDiagram(entries_dft + elems)

    Hf["E_h_ml"] = [pd_ml.get_e_above_hull(e) for e in entries_ml]
    Hf["E_h_dft"] = [pd_dft.get_e_above_hull(e) for e in entries_dft]

    Hf = Hf[Hf["comp_obj"].apply(lambda x: len(x) > 2)]
    Hf = Hf.sort_values("E_h_dft")
    Hf = Hf.reset_index(drop=True)

    # Hf["ml_sort"] = Hf["E_h_ml"].argsort().argsort()
    # Hf["dft_sort"] = np.argsort(Hf["E_h_dft"].values)

    # %%

    m = len(Hf)

    markers = [
        "o",
        # "v",
        "^",
        # "H",
        "D",
    ]

    for n, mk in zip([10, 20, 30], markers):
        null = np.zeros((m,))
        stab = np.zeros((m,))
        ids = Hf[Hf.E_h_dft <= n / 100].material_id.to_list()
        for i in range(m):
            n_found = np.sum(Hf.sort_values("E_h_ml")[:i].material_id.isin(ids))

            stab[i] = (n_found / i) / (len(ids) / len(Hf))

        print(n, stab[249])

        ax[k].plot(
            np.arange(m),
            stab,
            label=f"{n} meV",
            marker=mk,
            markevery=200,
            markersize=8 if mk == "D" else 10,
            # fillstyle="none",
            fillstyle="full",
            markerfacecolor="white",
            mew=2.5,
        )

    ax[k].set_xlim((0, 2400))

    ax[k].set_ylim((0, 10))

    ax[k].plot((0, m), (1, 1), "--", color="tab:grey", alpha=0.3)

    ax[k].set_xlabel("Number of Calculations")
    if k == 0:
        ax[k].set_ylabel("Enrichment Factor")

    ax[k].legend(title=rf"$\bf{{{el}-Zn-N}}$", frameon=False, loc="upper right")

plt.tight_layout()

plt.savefig(f"examples/plots/pdf/taata-enrich-{space}.pdf")

plt.show()


# %%
