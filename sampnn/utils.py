import random
from torch.utils.data.sampler import Sampler

class ClusterRandomSampler(Sampler):
    """
    Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches

    Arguments
    ---------
        data_source: torch.Dataset
            a Dataset to sample from. Should have a cluster_indices property
        batch_size: int
            a batch size that you would like to use later with Dataloader class
        shuffle: bool 
            whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)
