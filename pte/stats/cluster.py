"""Module for cluster-based statistics."""
from typing import Iterable, Optional
from matplotlib import pyplot as plt

import numpy as np
from numba import njit
from skimage import measure

import pte


def _null_distribution_2d(
    _data: np.ndarray, _alpha: float, _n_perm: int
) -> np.ndarray:
    """Calculate null distribution of clusters.

    Parameters
    ----------
    _data :  np.ndarray
        Data of three dimensions (first dimension is the number of
        measurements), e.g. shape: (n_subjects, n_freqs, n_times)
    _alpha : float
        Significance level (p-value)
    _n_perm : int
        No. of random permutations

    Returns
    -------
    null_distribution : np.ndarray
        Null distribution of shape (_n_perm, )
    """
    # loop through random permutation cycles
    null_distribution = np.zeros(_n_perm)
    for perm in range(_n_perm):
        print(f"Permutation: {perm}/{_n_perm}.")
        sign = np.random.choice(
            a=np.array([-1.0, 1.0]), size=_data.shape[0], replace=True
        ).reshape(_data.shape[0], 1, 1)
        _p_values = pte.stats.permutation_2d(
            x=_data.copy() * sign, y=0, n_perm=_n_perm, two_tailed=True
        )
        _labels, num_clusters = measure.label(
            _p_values <= _alpha, return_num=True, connectivity=2
        )

        max_p_sum = 0
        if num_clusters > 0:
            for i in range(num_clusters):
                _index_cluster = np.asarray(_labels == i + 1).nonzero()
                p_sum = np.sum(np.asarray(1 - _p_values)[_index_cluster])
                max_p_sum = max(p_sum, max_p_sum)
        null_distribution[perm] = max_p_sum
    return null_distribution


def cluster_2d(
    data: np.ndarray,
    alpha: float = 0.05,
    n_perm: int = 1000,
    only_max_cluster: bool = False,
) -> tuple[list, list]:
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_values :  numpy array
        Array of p-values. WARNING: MUST be one-dimensional
    alpha : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution
    only_max_cluster : bool, default = False
        Set to True to only return the most significant cluster.

    Returns
    -------
    cluster_pvals : list of float(s)
        List of p-values for each cluster
    clusters : list of numpy array(s)
        List of indices of each significant cluster
    """
    # Get 2D clusters
    p_values = pte.stats.permutation_2d(
        x=data, y=0, n_perm=n_perm, two_tailed=True
    )
    labels, num_clusters = measure.label(
        p_values <= alpha, return_num=True, connectivity=2
    )
    null_distr = _null_distribution_2d(data, alpha, n_perm)
    plt.hist(null_distr)
    plt.show(block=False)
    # Loop through clusters of p_val series or image
    clusters = []
    # Initialize empty list with specific data type for numba to work
    cluster_pvals = [np.float64(x) for x in range(0)]
    max_cluster_sum = 0  # Is only used if only_max_cluster
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        # index_cluster = np.where(labels == cluster_i + 1)[0]
        index_cluster = np.asarray(labels == cluster_i + 1)
        index_cluster = index_cluster.nonzero()
        p_cluster_sum = np.sum(np.asarray(1 - p_values)[index_cluster])
        print(f"Cluster value: {p_cluster_sum}")
        p_val = (n_perm - np.sum(p_cluster_sum >= null_distr) + 1) / n_perm
        if p_val <= alpha:
            clusters.append(index_cluster)
            cluster_pvals.append(p_val)
        if only_max_cluster:
            if p_cluster_sum > max_cluster_sum:
                clusters.clear()
                clusters.append(index_cluster)
                cluster_pvals = [p_val]
                max_cluster_sum = p_cluster_sum
    return cluster_pvals, clusters


def clusters_from_pvals(
    p_vals: np.ndarray,
    alpha: float,
    correction_method: str,
    n_perm: Optional[int],
) -> tuple[np.ndarray, int]:
    """Return significant clusters."""
    if np.where(p_vals <= alpha)[0].size > 1:
        signif = pte.stats.correct_pvals(
            p_vals=p_vals,
            alpha=alpha,
            correction_method=correction_method,
            n_perm=n_perm,
        )
        if signif.size > 1:
            clusters_raw = np.array(
                [1 if i in signif else 0 for i in range(len(p_vals))]
            )
            clusters, cluster_count = get_clusters(
                data=clusters_raw, min_cluster_size=1
            )
            return (clusters, cluster_count)
    return (np.ndarray([]), 0)


def get_clusters(data: Iterable, min_cluster_size: int = 1):
    """Cluster 1-D array of boolean values.

    Parameters
    ----------
    iterable : array-like of bool
        Array to be clustered.
    min_cluster_size : integer
        Minimum size of clusters to consider. Must be at least 1.

    Returns
    -------
    cluster_labels : np.array
        Array of shape (len(iterable), 1), where each value indicates the
        number of the cluster. Values are 0 if the item does not belong to
        a cluster
    cluster_count : int
        Number of detected cluster. Corresponds to the highest value in
        cluster_labels
    """
    min_cluster_size = max(min_cluster_size, 1)
    cluster_labels = np.zeros_like(data, dtype=int)
    cluster_count = 0
    cluster_len = 0
    for idx, item in enumerate(data):
        if item:
            cluster_len += 1
            cluster_labels[idx] = cluster_count + 1
        else:
            if cluster_len < min_cluster_size:
                cluster_labels[max(0, idx - cluster_len) : idx] = 0
            else:
                cluster_count += 1
            cluster_len = 0
    if cluster_len >= min_cluster_size:
        cluster_count += 1
    else:
        cluster_labels[min(-1, cluster_len) :] = 0
    return cluster_labels, cluster_count


@njit
def clusterwise_pval_numba(p_values, alpha, n_perm, only_max_cluster=False):
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_values :  numpy array
        Array of p-values. WARNING: MUST be one-dimensional
    alpha : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution
    only_max_cluster : bool, default = False
        Set to True to only return the most significant cluster.

    Returns
    -------
    cluster_pvals : list of float(s)
        List of p-values for each cluster
    clusters : list of numpy array(s)
        List of indices of each significant cluster
    """

    def _cluster(iterable):
        """Cluster 1-D array of boolean values.

        Parameters
        ----------
        iterable : array-like of bool
            Array to be clustered.

        Returns
        -------
        cluster_labels : np.array
            Array of shape (len(iterable), 1), where each value indicates the
            number of the cluster. Values are 0 if the item does not belong to
            a cluster
        cluster_count : int
            Number of detected cluster. Corresponds to the highest value in
            cluster_labels
        """
        cluster_labels = np.zeros((len(iterable), 1))
        cluster_count = 0
        cluster_len = 0
        for idx, item in enumerate(iterable):
            if item:
                cluster_labels[idx] = cluster_count + 1
                cluster_len += 1
            elif cluster_len == 0:
                pass
            else:
                cluster_len = 0
                cluster_count += 1
        if cluster_len >= 1:
            cluster_count += 1
        return cluster_labels, cluster_count

    def _null_distribution(_p_values, _alpha, _n_perm):
        """Calculate null distribution of clusters.

        Parameters
        ----------
        _p_values :  np.ndarray
            Array of p-values
        _alpha : float
            Significance level (p-value)
        _n_perm : int
            No. of random permutations

        Returns
        -------
        null_distribution : np.ndarray
            Null distribution of shape (_n_perm, )
        """
        # loop through random permutation cycles
        null_distribution = np.zeros(_n_perm)
        for i in range(_n_perm):
            r_per = np.random.randint(
                low=0, high=_p_values.shape[0], size=_p_values.shape[0]
            )
            pvals_perm = _p_values[r_per]
            labels_, n_clusters = _cluster(pvals_perm <= _alpha)

            cluster_ind = {}
            if n_clusters == 0:
                null_distribution[i] = 0
            else:
                p_sum = np.zeros(n_clusters)
                for ind in range(n_clusters):
                    cluster_ind[ind] = np.where(labels_ == ind + 1)[0]
                    p_sum[ind] = np.sum(
                        np.asarray(1 - pvals_perm)[cluster_ind[ind]]
                    )
                null_distribution[i] = np.max(p_sum)
        return null_distribution

    labels, num_clusters = _cluster(p_values <= alpha)

    null_distr = _null_distribution(p_values, alpha, n_perm)
    # Loop through clusters of p_val series or image
    clusters = []
    # Initialize empty list with specific data type for numba to work
    cluster_pvals = [np.float64(x) for x in range(0)]
    if only_max_cluster:
        max_cluster_sum = 0
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        index_cluster = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum = np.sum(np.asarray(1 - p_values)[index_cluster])
        p_val = (n_perm - np.sum(p_cluster_sum >= null_distr) + 1) / n_perm
        if p_val <= alpha:
            clusters.append(index_cluster)
            cluster_pvals.append(p_val)
        if only_max_cluster:
            if max_cluster_sum == 0 or p_cluster_sum > max_cluster_sum:
                clusters.clear()
                clusters.append(index_cluster)
                cluster_pvals = [p_val]
                max_cluster_sum = p_cluster_sum
    return cluster_pvals, clusters
