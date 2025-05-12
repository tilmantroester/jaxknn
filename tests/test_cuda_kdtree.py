import pytest
import numpy as np

from jaxknn.cuda_kdtree import knn_cuda as knn

from common import scipy_knn, UNIFORM_100_2D, UNIFORM_100_2D_BOX


@pytest.mark.parametrize(
    "points, queries, box_size, max_radius",
    [
        (UNIFORM_100_2D, None, None, np.inf),
        (UNIFORM_100_2D, UNIFORM_100_2D[:10], None, np.inf),
        (UNIFORM_100_2D, None, UNIFORM_100_2D_BOX, np.inf),
        (UNIFORM_100_2D, None, UNIFORM_100_2D_BOX, np.linspace(0.01, 0.2, UNIFORM_100_2D.shape[0])),
    ],
    ids=["all-knn", "10-queries", "periodic", "radius-per-query"]
)
def test_cuda_knn_interface(points, queries, box_size, max_radius):
    k = 9
    idx = knn(
        points=points, k=9, queries=queries, max_radius=max_radius, box_size=box_size
    )

    scipy_idx = scipy_knn(points=points, queries=queries,
                          k=k, box_size=box_size, max_radius=max_radius)
    # If n_points < k, scipy returns n_point, while cudakdtree returns -1
    scipy_idx[scipy_idx == scipy_idx.shape[0]] = -1

    assert np.all(idx == scipy_idx)
