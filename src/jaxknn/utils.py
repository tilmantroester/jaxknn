import numpy as np

import jax
import jax.numpy as np


def generate_uniform_random_points(n_point, n_dim):
    key = jax.random.key(42)
    points = jax.random.uniform(key=key, shape=(n_point, n_dim))

    return points, (1.0,)*n_dim


def generate_toy_2d_point_set():
    pos = jnp.array([
        [10, 15], [46, 63], [68, 21], [40, 33], [25, 54], [15, 43], [44, 58], [45, 40], [62, 69], [53, 67]
    ], dtype=jnp.float32)
    box_size = (70.0, 70.0)

    return pos, box_size
