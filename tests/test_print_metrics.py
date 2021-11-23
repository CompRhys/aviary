# %%

import numpy as np
import pandas as pd

from aviary.utils import print_metrics_classification, print_metrics_regression

xs, y_pred, y_true = pd.read_csv("tests/data/rand_regr.csv").to_numpy().T

y_binary, y_proba, y_clf = pd.read_csv("tests/data/rand_clf.csv").to_numpy().T
y_binary = y_binary.astype(int)
y_clf = y_clf.astype(int)
y_probs = np.tile(y_proba, (1, 2, 1))
y_probs[0, 1] = 1 - y_proba


def test_regression_metrics(capsys):
    print_metrics_regression(y_true, y_pred[None, :])
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.split("\n")
    assert len(lines) == 5
    assert out.startswith("Model Performance Metrics:\nR2 Score: ")
    assert lines[2].startswith("MAE: ")
    assert lines[3].startswith("RMSE: ")


def test_regression_metrics_ensemble(capsys):
    # simulate 2-model ensemble by duplicating predictions along 0-axis
    y_preds = np.tile(y_pred, (2, 1))
    print_metrics_regression(y_true, y_preds)
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.split("\n")
    assert len(lines) == 10
    assert out.startswith("Model Performance Metrics:\nR2 Score: ")
    assert lines[2].startswith("MAE: ") and "+/-" in lines[2]
    assert lines[3].startswith("RMSE: ") and "+/-" in lines[3]
    assert lines[5].startswith("Ensemble Performance Metrics:")


def test_classification_metrics(capsys):
    print_metrics_classification(y_binary, y_probs)
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.split("\n")
    assert len(lines) == 7


def test_classification_metrics_ensemble():
    y_probs = np.expand_dims(y_proba, axis=(0, 2))
    y_probs = np.tile(y_probs, (2, 1, 1))
    print_metrics_classification(y_binary, y_probs)
