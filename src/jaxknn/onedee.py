from functools import partial

import jax
import jax.numpy as jnp


def _check_knn_1d_args(points, k, queries, box_size):
    if k < 1:
        raise ValueError("k must be at least 1")
    if points.ndim != 1:
        raise ValueError("points must be a 1D array")
    if queries.ndim != 1:
        raise ValueError("queries must be a 1D array")
    if points.shape[0] < k:
        raise ValueError("points must have at least k elements")
    if queries.shape[0] < 1:
        raise ValueError("queries must have at least one element")
    if k > points.shape[0]:
        raise ValueError("k cannot be greater than the number of points")
    
    if box_size is not None and box_size <= 0:
        raise ValueError("box_size must be positive if provided")


def _knn_1d_bruteforce(points, k, queries=None, box_size=None):
    """
    Find the k nearest neighbors in 1D using brute force.
    Args:
        points: 1D array of points.
        k: Number of nearest neighbors to find.
        queries: 1D array of query points. If None, use points as queries.
        box_size: Size of the periodic box. If None, no periodic boundary is assumed.
    Returns:
        Indices of the k nearest neighbors for each query point.
    """
    if queries is None:
        queries = points
    _check_knn_1d_args(points, k, queries, box_size)
    
    d = jnp.abs(points[None, :] - queries[:, None])
    if box_size is not None:
        d = jnp.minimum(d, box_size - d)

    d, idx = jax.lax.top_k(-d, k)
    return idx


def _knn_1d_sorting(points, k, queries=None, box_size=None):
    """
    Find the k nearest neighbors in 1D using sorting.
    Args:
        points: 1D array of points.
        k: Number of nearest neighbors to find.
        queries: 1D array of query points. If None, use points as queries.
        box_size: Size of the periodic box. If None, no periodic boundary is assumed.
    Returns:
        Indices of the k nearest neighbors for each query point.
    """
    if queries is None:
        queries = points
    
    _check_knn_1d_args(points, k, queries, box_size)
    if 2*k + 1 > points.shape[0]:
        raise ValueError("2*k + 1 cannot be greater than the number of points")

    sort_idx = jnp.argsort(points)

    sorted_points = points[sort_idx]

    # Need to map back to unsorted order
    query_idx = jnp.searchsorted(sorted_points, queries)

    n = points.shape[0]
    neighbor_idx = query_idx[:, None] + jnp.arange(-k, k+1)[None, :]

    if box_size is not None:
        neighbor_idx = jnp.where(neighbor_idx >= n, neighbor_idx - n, neighbor_idx)
        neighbor_idx = jnp.where(neighbor_idx < 0, neighbor_idx + n, neighbor_idx)
        d = jnp.abs(sorted_points[neighbor_idx] - queries[:, None])
        d = jnp.minimum(d, box_size - d)
    else:
        # We need a fixed 2k+1 neighbor indices, so for the non-periodic
        # case, we clip the the 2k+1 window to the ends of the array
        neighbor_idx = jnp.where(
            query_idx[:, None] < n - k,
            jnp.where(
                query_idx[:, None] >= k,
                query_idx[:, None] + jnp.arange(-k, k)[None, :],
                jnp.arange(2*k)[None, :]
            ),
            n - (jnp.arange(2*k)[None, :] + 1)
        )
        d = jnp.abs(sorted_points[neighbor_idx] - queries[:, None])
    
    # Sort distances and get indices
    d, idx = jax.lax.top_k(-d, k)
    idx = jnp.take_along_axis(neighbor_idx, idx, axis=1)
    idx = sort_idx[idx]

    return idx


partial(jax.jit, static_argnames=["k", "bruteforce"])
def knn_1d(points, k, queries=None, bruteforce=False, box_size=None):
    """
    Find the k nearest neighbors in 1D.
    Args:
        points: 1D array of points.
        k: Number of nearest neighbors to find.
        queries: 1D array of query points. If None, use points as queries.
        bruteforce: If True, use brute force method. Otherwise, use sorting method. For small datasets (about <10k), brute force is faster.
        box_size: Size of the periodic box. If None, no periodic boundary is assumed.
    Returns:
        Indices of the k nearest neighbors for each query point.
    """
    if bruteforce:
        return _knn_1d_bruteforce(points, k, queries, box_size)
    else:
        return _knn_1d_sorting(points, k, queries, box_size)