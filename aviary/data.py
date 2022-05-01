from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
from torch import Tensor


@dataclass
class InMemoryDataLoader:
    """In-memory DataLoader using array/tensor slicing instead to generate whole
    batches at once instead of sample by sample.

    Args:
        *tensors: List of arrays or tensors. Must all have the same length @ dim 0.
        batch_size (int, optional): Defaults to 32.
        shuffle (bool, optional): If True, shuffle the data *in-place* whenever an
            iterator is created from this object. Defaults to False.
    """

    tensors: list[Tensor]
    batch_size: int = 32
    shuffle: bool = False
    collate_fn: Callable[[Any], tuple[Tensor]] = tuple

    def __post_init__(self):
        self.dataset_len = len(self.tensors[0])
        if not all(len(t) == self.dataset_len for t in self.tensors):
            raise ValueError("All tensors must have the same length in dim 0")

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        if self.shuffle:
            shuffle_idx = torch.randperm(self.dataset_len)
            self.tensors = [t[shuffle_idx] for t in self.tensors]
        self.idx = 0
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        if self.idx >= self.dataset_len:
            raise StopIteration
        slices = (t[self.idx : self.idx + self.batch_size] for t in self.tensors)
        batch = self.collate_fn(*slices)

        self.idx += self.batch_size
        return batch

    def __len__(self) -> int:
        """Get the number of batches in this dataloader"""
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        return n_batches + bool(remainder)
