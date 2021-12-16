"""Module for permutation testing."""

from numba import njit
import numpy as np


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
        z = np.abs(np.mean(x) - np.mean(y))
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
    else:
        z = np.mean(x) - np.mean(y)
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(pS[:half]) - np.mean(pS[half:])
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)
