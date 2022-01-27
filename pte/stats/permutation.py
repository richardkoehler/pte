"""Module for permutation testing."""

from numba import njit
import numpy as np


@njit
def permutation_2d(x, y, n_perm=1000, two_tailed=True):
    """Perform permutation test with one-sample distribution.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int or float
        Baseline against which to check for statistical significane
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """

    def permutation_pval(x, y, n_perm, two_tailed):
        zeroed = x - y
        p = np.empty(n_perm)
        z = np.mean(zeroed)
        if two_tailed:
            z = np.abs(z)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(
                a=np.array([-1.0, 1.0]), size=len(x), replace=True
            )
            val_perm = np.mean(zeroed * sign)
            if two_tailed:
                val_perm = np.abs(val_perm)
            p[i] = val_perm
        # Return p-value
        return z, (np.sum(p >= z) + 1) / (n_perm + 1)

    p_vals = np.empty((x.shape[1], x.shape[2]))
    for i in np.arange(x.shape[1]):
        for j in np.arange(x.shape[2]):
            _, p = permutation_pval(x[:, i, j], y, n_perm, two_tailed)
            p_vals[i, j] = p
    return p_vals


@njit
def permutation_onesample(x, y, n_perm=10000, two_tailed=True):
    """Perform permutation test with one-sample distribution.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int or float
        Baseline against which to check for statistical significane
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """
    zeroed = x - y
    p = np.empty(n_perm)
    z = np.mean(zeroed)
    if two_tailed:
        z = np.abs(z)
    # Run the simulation n_perm times
    for i in np.arange(n_perm):
        sign = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(x), replace=True
        )
        val_perm = np.mean(zeroed * sign)
        if two_tailed:
            val_perm = np.abs(val_perm)
        p[i] = val_perm
    # Return p-value
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_twosample(x, y, n_perm=10000, two_tailed=True):
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like
        Second distribution
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means
    float
        P-value of permutation test
    """
    if two_tailed is True:
        zeroed = np.abs(np.mean(x) - np.mean(y))
        data = np.concatenate((x, y), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(data[:half]) - np.mean(data[half:]))
    else:
        zeroed = np.mean(x) - np.mean(y)
        data = np.concatenate((x, y), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(data[:half]) - np.mean(data[half:])
    return zeroed, (np.sum(p >= zeroed) + 1) / (n_perm + 1)
