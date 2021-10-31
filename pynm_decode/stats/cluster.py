"""Module for cluster-based statistics."""

import numpy as np
from numba import njit


@njit
def clusterwise_pval_numba(p_arr, p_sig, n_perm):
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_arr :  array-like
        Array of p-values. WARNING: MUST be one-dimensional
    p_sig : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution

    Returns
    -------
    p : list of floats
        List of p-values for each cluster
    p_min_index : list of numpy array
        List of indices of each significant cluster
    """

    def cluster(iterable):
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

    def calculate_null_distribution(p_arr_, p_sig_, n_perm_):
        """Calculate null distribution of clusters.

        Parameters
        ----------
        p_arr_ :  np.ndarray
            Array of p-values
        p_sig_ : float
            Significance level (p-value)
        n_perm_ : int
            No. of random permutations

        Returns
        -------
        r_per_arr : np.ndarray
            Null distribution of shape (n_perm_)
        """
        # loop through random permutation cycles
        r_per_arr = np.zeros(n_perm_)
        for i in range(n_perm_):
            r_per = np.random.randint(
                low=0, high=p_arr_.shape[0], size=p_arr_.shape[0]
            )
            labels_, n_clusters = cluster(p_arr_[r_per] <= p_sig_)

            cluster_ind = {}
            if n_clusters == 0:
                r_per_arr[i] = 0
            else:
                p_sum = np.zeros(n_clusters)
                for ind in range(n_clusters):
                    cluster_ind[ind] = np.where(labels_ == ind + 1)[0]
                    p_sum[ind] = np.sum(
                        np.asarray(1 - p_arr_[r_per])[cluster_ind[ind]]
                    )
                r_per_arr[i] = np.max(p_sum)
        return r_per_arr

    labels, num_clusters = cluster(p_arr <= p_sig)

    null_distr = calculate_null_distribution(p_arr, p_sig, n_perm)
    # Loop through clusters of p_val series or image
    clusters = []
    # Initialize empty list with specific data type for numba to work
    p_vals = [np.float64(x) for x in range(0)]
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        index_cluster = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum = np.sum(np.asarray(1 - p_arr)[index_cluster])
        p_val = (n_perm - np.sum(p_cluster_sum >= null_distr) + 1) / n_perm
        if p_val <= p_sig:
            clusters.append(index_cluster)
            p_vals.append(p_val)

    return p_vals, clusters
