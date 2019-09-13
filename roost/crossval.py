def cv_ensemble(fold_id, dataset, ensemble_folds, fea_len):
    """
    Divide the dataset into X folds.

    Keeping one fold as a hold-out set train a 
    model on the next fold for a given number of epochs.

    using the hold-out set keep the best 
    performing model over the whole training period.
    """

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}

    total = len(dataset)
    splits = k_fold_split(ensemble_folds, total)

    model_dir = "models/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for run_id, (train, val) in enumerate(splits):

        device, model, criterion, optimizer, normalizer = init_model(fea_len)

        train_subset = torch.utils.data.Subset(dataset, train)
        val_subset = torch.utils.data.Subset(dataset, val)

        train_generator = DataLoader(train_subset, **params)
        val_generator = DataLoader(val_subset, **params)

        _, sample_target, _ = collate_batch(train_subset)
        normalizer.fit(sample_target)

        experiment(model_dir, fold_id, run_id, args, train_generator, val_generator, 
            model, optimizer, criterion, normalizer)
     

    test_ensemble(model_dir, fold_id, ensemble_folds, test_set, orig_atom_fea_len)




def nested_cv(cv_folds=5):
    """
    Divide the total dataset into X folds.

    Keeping one fold as a hold out set train 
    an ensemble of models on the remaining data.

    Iterate such that each fold is used as the 
    hold out set once and return the cross validation error.
    """

    dataset = CompositionData(args.data_dir, seed=43)

    orig_atom_fea_len = dataset.atom_fea_dim

    total = len(dataset)

    splits = k_fold_split(cv_folds, total)

    for fold_id, (training, hold_out) in enumerate(splits):
        training_set = torch.utils.data.Subset(dataset, training)
        hold_out_set = torch.utils.data.Subset(dataset, hold_out)

        ensemble_folds = 10
        cv_ensemble(fold_id, training_set, ensemble_folds, orig_atom_fea_len)

        break


def partitions(number, k):
    """
    Distribution of the folds allowing for cases where 
    the folds do not divide evenly

    Inputs
    --------
    k: int
        The number of folds to split the data into
    number: int
        The number of datapoints in the dataset
    """
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions



def get_indices(n_splits, points):
    """
    Indices of the set test

    Inputs
    --------
    n_splits: int
        The number of folds to split the data into
    points: int
        The number of datapoints in the dataset
    """
    fold_sizes = partitions(points, n_splits)
    indices = np.arange(points).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])



def k_fold_split(n_splits = 3, points = 3001):
    """
    Generates folds for cross validation

    Inputs
    --------
    n_splits: int
        The number of folds to split the data into
    points: int
        The number of datapoints in the dataset

    """
    indices = np.arange(points).astype(int)
    for test_idx in get_indices(n_splits, points):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx