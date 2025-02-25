import pytest
import numpy as np

from jaxknn.cuda_kdtree import knn_cuda as knn

from common import scipy_knn, UNIFORM_100_2D, UNIFORM_100_2D_BOX


@pytest.mark.parametrize(
    "points, queries, box_size",
    [
        (UNIFORM_100_2D, None, None),
        (UNIFORM_100_2D, UNIFORM_100_2D[:10], None),
        (UNIFORM_100_2D, None, UNIFORM_100_2D_BOX),
    ],
    ids=["all-knn", "10-queries", "periodic"]
)
def test_cuda_knn_interface(points, queries, box_size):
    k = 9
    idx = knn(
        points=points, k=9, queries=queries, box_size=box_size
    )

    scipy_idx = scipy_knn(points=points, queries=queries,
                          k=k, box_size=box_size, max_radius=np.inf)

    assert np.all(idx == scipy_idx)
