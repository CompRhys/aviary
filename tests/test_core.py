import numpy as np

from aviary.core import np_one_hot, np_softmax


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
