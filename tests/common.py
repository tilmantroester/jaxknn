from collections.abc import Iterable

import jax
import jax.numpy as jnp

import numpy as np
import scipy.spatial


def uniform_random_points(n_dim, n_point):
    key = jax.random.key(42)
    points = jax.random.uniform(key=key, shape=(n_point, n_dim))

    return points, (1.0,)*n_dim


def toy_2d_point_set():
    pos = jnp.array([
        [10, 15], [46, 63], [68, 21], [40, 33], [25, 54], [15, 43], [44, 58], [45, 40], [62, 69], [53, 67]
    ], dtype=jnp.float32)
    box_size = (70.0, 70.0)

    return pos, box_size


def scipy_knn(points, queries, k, box_size, max_radius):
    kdtree = scipy.spatial.KDTree(points, boxsize=box_size)
    if queries is None:
        queries = points
    if isinstance(max_radius, Iterable):
        scipy_idx = []
        for query, r in zip(points, max_radius):
            _, idx = kdtree.query(query, k=k, distance_upper_bound=r)
            scipy_idx.append(idx)
        scipy_idx = np.asarray(scipy_idx)
    else:
        _, scipy_idx = kdtree.query(queries, k=k, distance_upper_bound=max_radius)
    return scipy_idx


UNIFORM_100_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=100)
UNIFORM_1000_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=1000)

UNIFORM_100_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=100)
UNIFORM_1000_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=1000)

TOY_10_2D, TOY_10_2D_BOX = toy_2d_point_set()