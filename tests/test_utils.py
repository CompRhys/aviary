import json

import numpy as np
import pandas as pd
import pytest
import torch

from aviary.utils import get_element_embedding, get_metrics, get_sym_embedding


@pytest.fixture
def temp_element_embedding(tmp_path):
    embedding_data = {
        "H": [1.0, 2.0],
        "He": [3.0, 4.0],
        "Li": [5.0, 6.0],
    }
    path = tmp_path / "test_elem_embedding.json"
    with open(path, "w") as f:
        json.dump(embedding_data, f)
    return str(path)


@pytest.fixture
def temp_sym_embedding(tmp_path):
    embedding_data = {
        "1": {"a": [1.0, 2.0], "b": [3.0, 4.0]},
        "2": {"c": [5.0, 6.0]},
    }
    path = tmp_path / "test_sym_embedding.json"
    with open(path, "w") as f:
        json.dump(embedding_data, f)
    return str(path)


def test_get_element_embedding_custom(temp_element_embedding):
    embedding = get_element_embedding(temp_element_embedding)
    assert isinstance(embedding, torch.nn.Embedding)
    assert embedding.weight.shape == (3 + 1, 2)  # max_Z + 1, embedding_dim
    assert torch.allclose(embedding.weight[1], torch.tensor([1.0, 2.0]))  # H
    assert torch.allclose(embedding.weight[2], torch.tensor([3.0, 4.0]))  # He


def test_get_element_embedding_builtin():
    embedding = get_element_embedding("matscholar200")
    assert isinstance(embedding, torch.nn.Embedding)
    assert embedding.weight.shape[1] == 200


def test_get_element_embedding_invalid():
    with pytest.raises(ValueError, match="Invalid element embedding: invalid_embedding"):
        get_element_embedding("invalid_embedding")


def test_get_sym_embedding_custom(temp_sym_embedding):
    embedding = get_sym_embedding(temp_sym_embedding)
    assert isinstance(embedding, torch.nn.Embedding)
    assert embedding.weight.shape == (3, 2)  # total features, embedding_dim
    assert torch.allclose(embedding.weight[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(embedding.weight[1], torch.tensor([3.0, 4.0]))


def test_get_sym_embedding_builtin():
    embedding = get_sym_embedding("bra-alg-off")
    assert isinstance(embedding, torch.nn.Embedding)
    assert isinstance(embedding.weight, torch.Tensor)


def test_get_sym_embedding_invalid():
    with pytest.raises(ValueError, match="Invalid symmetry embedding: invalid_embedding"):
        get_sym_embedding("invalid_embedding")


def test_regression_metrics():
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    metrics = get_metrics(targets, predictions, "regression")

    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] == pytest.approx(0.1, abs=1e-4)
    assert metrics["RMSE"] == pytest.approx(0.1, abs=1e-4)
    assert metrics["R2"] == pytest.approx(0.995, abs=1e-4)


def test_classification_metrics():
    targets = np.array([0, 1, 0, 1, 0])
    # Probabilities for class 0 and 1
    predictions = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])

    metrics = get_metrics(targets, predictions, "classification")

    assert set(metrics.keys()) == {"accuracy", "balanced_accuracy", "F1", "ROCAUC"}
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["F1"] == 1.0
    assert metrics["ROCAUC"] == 1.0


def test_nan_handling():
    targets = np.array([1.0, np.nan, 3.0, 4.0])
    predictions = np.array([1.1, 2.1, np.nan, 4.1])

    metrics = get_metrics(targets, predictions, "regression")
    assert not np.isnan(metrics["MAE"])
    assert not np.isnan(metrics["RMSE"])
    assert not np.isnan(metrics["R2"])


def test_pandas_input():
    targets = pd.Series([1.0, 2.0, 3.0])
    predictions = pd.Series([1.1, 2.1, 3.1])

    metrics = get_metrics(targets, predictions, "regression")
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_precision():
    targets = np.array([1.0, 2.0, 3.0])
    predictions = np.array([1.12345, 2.12345, 3.12345])

    metrics = get_metrics(targets, predictions, "regression", prec=2)
    assert all(len(str(v).split(".")[-1]) <= 2 for v in metrics.values())


def test_invalid_type():
    targets = np.array([1.0, 2.0])
    predictions = np.array([1.1, 2.1])

    with pytest.raises(ValueError, match="Invalid task type: invalid_type"):
        get_metrics(targets, predictions, "invalid_type")


def test_mismatched_shapes():
    targets = np.array([0, 1, 0])
    predictions = np.array([[0.9, 0.1], [0.1, 0.9]])  # Wrong shape

    with pytest.raises(ValueError):  # noqa: PT011
        get_metrics(targets, predictions, "classification")
