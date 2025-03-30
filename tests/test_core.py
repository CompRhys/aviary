import numpy as np
import pytest
import torch

from aviary.core import masked_mean, masked_std, np_one_hot, np_softmax


def test_np_one_hot():
    assert np.allclose(np_one_hot(np.arange(3)), np.eye(3))

    # test n_classes kwarg
    out = np_one_hot(np.arange(3), n_classes=5)
    expected = np.eye(5)[np.arange(3)]
    assert np.allclose(out, expected)


def test_np_softmax():
    x1 = np.array([[0, 1, 2], [3, 4, 5]])
    expected = np.array([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]])
    assert np.allclose(np_softmax(x1), expected, atol=1e-4)

    for xi in np.random.rand(3, 2, 3):
        for axis in (0, 1):
            # test numbers in softmaxed dimension all sum to 1
            out = np_softmax(xi, axis=axis)
            assert np.allclose(out.sum(axis=axis), 1)


def test_masked_mean():
    # test 1d tensor
    x1 = torch.arange(5).float()
    mask1 = torch.tensor([0, 1, 0, 1, 0]).bool()
    assert masked_mean(x1, mask1) == 2

    assert masked_mean(x1, mask1) == sum(x1 * mask1) / sum(mask1)

    # test 2d tensor
    x2 = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    mask2 = torch.tensor([[1, 1, 0], [0, 0, 1]]).bool()

    assert masked_mean(x2, mask2) == pytest.approx([1, 2, 6])
    assert masked_mean(x2, mask2, dim=1) == pytest.approx([1.5, 6])


def test_masked_std():
    # test 1d tensor
    x1 = torch.arange(5).float()
    mask1 = torch.tensor([0, 1, 0, 1, 0]).bool()
    assert masked_std(x1, mask1) == 1

    # test 2d tensor
    x2 = torch.tensor([[1, 1, 1], [2, 2, 4]]).float()
    mask2 = torch.tensor([[0, 1, 0], [1, 0, 1]]).bool()
    assert masked_std(x2, mask2, dim=0) == pytest.approx([0, 0, 0], abs=1e-4)
    assert masked_std(x2, mask2, dim=1) == pytest.approx([0, 1], abs=1e-4)

    # test against explicit calculation
    rand_floats = torch.rand(3, 4, 5)
    rand_masks = torch.randint(0, 2, (3, 4, 5)).bool()
    for xi, mask in zip(rand_floats, rand_masks, strict=False):
        for dim in (0, 1):
            out = masked_std(xi, mask, dim=dim)
            xi_nan = torch.where(mask, xi, torch.tensor(float("nan")))
            mean = xi_nan.nanmean(dim=dim)
            std = (xi_nan - mean.unsqueeze(dim=dim)).pow(2).nanmean(dim=dim).sqrt()

            assert out == pytest.approx(std, abs=1e-4, nan_ok=True)
