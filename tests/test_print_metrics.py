import numpy as np

from aviary.utils import print_metrics_classification, print_metrics_regression

# generate random data to test print functions
rng = np.random.RandomState(42)

# Generate reg data
xs = rng.rand(100)
y_pred = xs + 0.1 * rng.normal(size=100)
y_true = xs + 0.1 * rng.normal(size=100)

# Generate clf data
y_binary = rng.choice([0, 1], (100))
y_proba = np.clip(y_binary - 0.1 * rng.normal(scale=5, size=(100)), 0.1, 0.9)

# NOTE binary clf is handled as a multi-class clf problem therefore we need
# to add another prediction dimension to accommodate the negative class
y_probs = np.expand_dims(y_proba, axis=(0, 2))
y_probs = np.tile(y_probs, (1, 1, 2))
y_probs[0, :, 1] = 1 - y_proba


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
    assert len(lines) == 8
    assert out.startswith("\nModel Performance Metrics:\nAccuracy")
    assert lines[3].startswith("ROC-AUC")
    assert lines[4].startswith("Weighted Precision")
    assert lines[5].startswith("Weighted Recall")
    assert lines[6].startswith("Weighted F-score")


def test_classification_metrics_ensemble(capsys):
    y_probs_ens = np.tile(y_probs, (2, 1, 1))
    print_metrics_classification(y_binary, y_probs_ens)
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.split("\n")
    assert len(lines) == 15
    assert out.startswith("\nModel Performance Metrics:\nAccuracy")
    assert lines[3].startswith("ROC-AUC") and "+/-" in lines[3]
    assert lines[4].startswith("Weighted Precision") and "+/-" in lines[4]
    assert lines[5].startswith("Weighted Recall") and "+/-" in lines[5]
    assert lines[6].startswith("Weighted F-score") and "+/-" in lines[6]
    assert lines[8].startswith("Ensemble Performance Metrics:")
