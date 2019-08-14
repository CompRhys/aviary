
def main():
    """
    generate scripts for submitting jobs
    """

    size = np.logspace(0, 2.7, 9, dtype=int)[::-1]

    datasets = ["data/datasets/oqmd-form-enthalpy.csv", "data/datasets/expt-non-metals.csv",
                "data/datasets/mp-non-metals.csv"]

    oqmd

    atom_fea_len=64, 
    batch_size=128, 
    clr=1, 
    data_path='data/datasets/oqmd-form-enthalpy.csv', 
    ensemble=1,
    fold_id=8, 
    learning_rate=5e-04, 
    loss='L2', 
    momentum=0.9, 
    n_graph=3, 
    optim='Adam', 
    resume=False, 
    run_id=0, 
    seed=42, 
    sub_sample=1, 
    test_size=0.2, 
    transfer=None, 
    weight_decay=1e-05, 

if __name__ == "__main__":
    main()
    pass