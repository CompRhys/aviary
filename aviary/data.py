from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from torch import Tensor


@dataclass
class InMemoryDataLoader:
    """In-memory DataLoader using array/tensor slicing to generate whole batches at
    once instead of sample-by-sample.
    Source: https://discuss.pytorch.org/t/27014/6

    Args:
        *tensors: List of arrays or tensors. Must all have the same length in dimension 0.
        batch_size (int, optional): Defaults to 32.
        shuffle (bool, optional): If True, shuffle the data *in-place* whenever an
            iterator is created from this object. Defaults to False.
        collate_fn (Callable, optional): Should accept variadic list of tensors and
            output a minibatch of data ready for model consumption. Defaults to tuple().
    """

    tensors: list[Tensor]
    batch_size: int = 32
    shuffle: bool = False
    collate_fn: Callable = tuple

    def __post_init__(self):
        self.dataset_len = len(self.tensors[0])
        if not all(len(t) == self.dataset_len for t in self.tensors):
            raise ValueError("All tensors must have the same length in dim 0")

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        self.indices = np.random.permutation(self.dataset_len) if self.shuffle else None
        self.current_idx = 0
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        start_idx = self.current_idx
        if start_idx >= self.dataset_len:
            raise StopIteration

        end_idx = start_idx + self.batch_size

        if self.indices is None:  # shuffle=False
            slices = (t[start_idx:end_idx] for t in self.tensors)
        else:
            idx = self.indices[start_idx:end_idx]
            slices = (t[idx] for t in self.tensors)

        batch = self.collate_fn(*slices)

        self.current_idx += self.batch_size
        return batch

    def __len__(self) -> int:
        """Get the number of batches in this data loader."""
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        return n_batches + bool(remainder)
