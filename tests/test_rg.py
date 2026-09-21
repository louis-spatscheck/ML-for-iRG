import numpy as np

from invrg.rg import majority_rule


def test_majority_rule_blocks():
    config = np.array([[[1, 1, -1, -1],
                        [1, 1, -1, 1],
                        [-1, -1, 1, 1],
                        [-1, 1, 1, 1]]], dtype=float)
    result = majority_rule(config, 4)
    # block sums: 4, -2 / -2, 4
    assert result.shape == (1, 2, 2)
    assert result[0].tolist() == [[1, -1], [-1, 1]]


def test_majority_rule_ties_are_random_spins():
    config = np.tile(np.array([[1, -1], [1, -1]], dtype=float), (200, 1, 1))  # every block sums to 0
    result = majority_rule(config, 2)
    assert set(np.unique(result)) == {-1.0, 1.0}
    assert 0.3 < np.mean(result == 1) < 0.7
