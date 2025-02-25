from functools import partial

import numpy as np

import scipy.spatial

import jax
import jax.numpy as jnp


def scipy_get_k_nearest_neighbor_idx(points, queries, k,
                                     distance_upper_bound=np.inf,
                                     boxsize=None, workers=-1, leafsize=10):
    kd_tree = scipy.spatial.cKDTree(
        data=points, boxsize=boxsize, leafsize=leafsize)
    distances, idx = kd_tree.query(
        x=queries,
        k=int(k),
        workers=workers,
        distance_upper_bound=distance_upper_bound
    )
    return distances.astype(points.dtype), idx.astype(np.int32)


@partial(jax.jit, static_argnames=[
    "k", "distance_upper_bound", "boxsize", "workers", "leafsize"]
)
def jax_callback_get_k_nearest_neighbor_idx(points, k, queries=None,
                                            distance_upper_bound=np.inf,
                                            boxsize=None, workers=-1,
                                            leafsize=10):
    if queries is None:
        queries = points
    shape = (jnp.shape(queries)[0], k)
    distance_type = jax.ShapeDtypeStruct(shape, points.dtype)
    idx_type = jax.ShapeDtypeStruct(shape, jnp.int32)
    return jax.pure_callback(
        scipy_get_k_nearest_neighbor_idx,
        (distance_type, idx_type),
        points, queries, k, distance_upper_bound, boxsize, workers, leafsize
    )


def knn_scipy(points, k, queries=None, max_radius=np.inf, box_size=None, **kwargs):
    _, idx = jax_callback_get_k_nearest_neighbor_idx(
        points=points, k=k, queries=queries,
        distance_upper_bound=max_radius, boxsize=box_size, **kwargs
    )
    return idx
