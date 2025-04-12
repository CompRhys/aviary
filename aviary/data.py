from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self) -> None:
        """Initialize Normalizer with mean 0 and std 1."""
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor: Tensor, dim: int = 0, keepdim: bool = False) -> None:
        """Compute the mean and standard deviation of the given tensor.

        Args:
            tensor (Tensor): Tensor to determine the mean and standard deviation over.
            dim (int, optional): Which dimension to take mean and standard deviation
                over. Defaults to 0.
            keepdim (bool, optional): Whether to keep the reduced dimension in Tensor.
                Defaults to False.
        """
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor: Tensor) -> Tensor:
        """Normalize a Tensor.

        Args:
            tensor (Tensor): Tensor to be normalized

        Returns:
            Tensor: Normalized Tensor
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: Tensor) -> Tensor:
        """Restore normalized Tensor to original.

        Args:
            normed_tensor (Tensor): Tensor to be restored

        Returns:
            Tensor: Restored Tensor
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict[str, Tensor]:
        """Get Normalizer parameters mean and std.

        Returns:
            dict[str, Tensor]: Dictionary storing Normalizer parameters.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Overwrite Normalizer parameters given a new state_dict.

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.
        """
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Tensor]) -> Self:
        """Create a new Normalizer given a state_dict.

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.

        Returns:
            Normalizer
        """
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


@dataclass
class InMemoryDataLoader:
    """In-memory DataLoader using array/tensor slicing to generate whole batches at
    once instead of sample-by-sample.
    Source: https://discuss.pytorch.org/t/27014/6.

    Args:
        *tensors: List of arrays or tensors. Must all have the same length in
            dimension 0.
        collate_fn (Callable): Should accept variadic list of tensors and
            output a minibatch of data ready for model consumption.
        batch_size (int, optional): Usually 64, 128 or 256. Can be larger for test set
            loaders to speedup inference. Defaults to 64.
        shuffle (bool, optional): If True, shuffle the data *in-place* whenever an
            iterator is created from this object. Defaults to False.
    """

    # each item must be indexable (usually torch.tensor, np.array or pd.Series)
    tensors: list[Tensor | np.ndarray]
    collate_fn: Callable
    batch_size: int = 64
    shuffle: bool = False

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
