"""Module for permutation testing."""

from numba import njit
import numpy as np


@njit
def permutation_numba_onesample(x, y, n_perm, two_tailed=True):
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
    if two_tailed is True:
        zeroed = x - y
        z = np.abs(np.mean(zeroed))
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(
                a=np.array([-1.0, 1.0]), size=len(x), replace=True
            )
            p[i] = np.abs(np.mean(zeroed * sign))
    else:
        zeroed = x - y
        z = np.mean(zeroed)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(
                a=np.array([-1.0, 1.0]), size=len(x), replace=True
            )
            p[i] = np.mean(zeroed * sign)
        # Return p-value
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_numba_twosample(x, y, n_perm, two_tailed=True):
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


def permutation_test(x, y, n_perm, two_tailed=True):
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like or int or float
        Second distribution in the case of two-sampled test or baseline in the
        case of one-sampled test
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means or difference of single
        distribution mean from baseline
    float
        P-value of permutation test
    """
    if isinstance(y, (int, float)):
        # Perform one-sample permutation test
        if two_tailed:
            # Perform two-tailed permutation test
            zeroed = x - y
            z = np.abs(np.mean(zeroed))
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(n_perm):
                sign = np.random.choice(
                    a=np.array([-1.0, 1.0]), size=len(x), replace=True
                )
                p[i] = np.abs(np.mean(zeroed * sign))
        else:
            # Perform one-tailed permutation test
            zeroed = x - y
            z = np.mean(zeroed)
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(n_perm):
                sign = np.random.choice(
                    a=np.array([-1.0, 1.0]), size=len(x), replace=True
                )
                p[i] = np.mean(zeroed * sign)
    else:
        # Perform two-sample permutation test
        if two_tailed:
            # Perform two-tailed permutation test
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
            # Perform one-tailed permutation test
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
        # Return p-value
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_onesample_onetailed(x, y, n_perm):
    """"""
    # Perform one-tailed permutation test
    zeroed = x - y
    z = np.mean(zeroed)
    p = np.empty(n_perm)
    # Run the simulation n_perm times
    for i in np.arange(n_perm):
        sign = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(x), replace=True
        )
        p[i] = np.mean(zeroed * sign)
    return (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_onesample_twotailed(x, baseline, n_perm):
    """"""
    ## Initialize and pre-allocate
    zeroed = x - baseline
    sample_mean = np.abs(np.mean(zeroed))
    ## Run the simulation n_perm times
    z = np.empty(n_perm)
    for i in np.arange(n_perm):
        #  1. take n random draws from {-1, 1}, where len(x) is the length of
        #     the data to be tested
        mn = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(x), replace=True
        )
        #  2. assign the signs to the data and put them in a temporary variable
        flipped = zeroed * mn
        #  3. save the new data in an array
        z[i] = np.abs(np.mean(flipped))
    return (np.sum(z >= sample_mean) + 1) / (n_perm + 1)


@njit
def permutation_twosample_twotailed(x, y, n_perm):
    """Perform two-tailed permutation test of two distributions.

    Parameters
    ----------
    x : np array
        First distribution
    y : np array
        Second distribution
    n_perm : int
        Number of permutations

    Returns
    -------
    float
        estimated ground truth, here abs difference of distribution means
    float
        p-value of permutation test
    """
    # Compute ground truth difference
    sample_diff = np.abs(np.mean(x) - np.mean(y))
    half = len(x)
    # Initialize permutation
    pS = np.concatenate((x, y), axis=0)
    z = np.empty(n_perm)
    # Permutation loop
    for i in np.arange(0, n_perm):
        # Shuffle the data
        np.random.shuffle(pS)
        # Compute permuted absolute difference of the two sampled distributions
        z[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
    # Return p-value
    return (len(np.where(z >= sample_diff)[0]) + 1) / (n_perm + 1)
