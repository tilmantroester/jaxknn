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
    if queries is None:
        queries = points
    kdtree = scipy.spatial.KDTree(points, boxsize=box_size)
    _, scipy_idx = kdtree.query(queries, k=k, distance_upper_bound=max_radius)
    return scipy_idx


UNIFORM_100_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=100)
UNIFORM_1000_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=1000)

UNIFORM_100_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=100)
UNIFORM_1000_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=1000)

TOY_10_2D, TOY_10_2D_BOX = toy_2d_point_set()