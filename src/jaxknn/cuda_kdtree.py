import numpy as np

from cudakdtree_jax_binding.cudakdtree_interface import kdtree_call
from cudakdtree_jax_binding.cudakdtree_interface import TraversalMode as _TraversalMode
from cudakdtree_jax_binding.cudakdtree_interface import CandidateList as _CandidateList


TraversalMode = _TraversalMode
CandidateList = _CandidateList

# HEAP_CANDIDATE_LIST_THRESHOLD = 128

def knn_cuda(points, k, queries=None, max_radius=np.inf, box_size=None,
             traversal_mode=TraversalMode.stack_free_bounds_tracking,
             candidate_list=CandidateList.fixed_list):
    """Computes k-nearest neighbors using cudaKDTree.

    Arguments
    ---------
    points: array
        Points used for knn search.
    k: int
        Number of neighbors.
    queries: array, optional
        Query points. If `None`, use `points` as queries.
    max_radius: float, optional
        Maximum search radius. Default `numpy.inf`.
    box_size: tuple, optional
        Dimensions with toroidal topology. If `box_size[i] > 0`, the 
        box will wrap around with period `box_size[i]`. If `box_size[i] <= 0`,
        dimension `i` is treated as non-periodic. Default `None`.
    
    Returns
    -------
    idx: array
        Indices of the neighbors. Shape `(queries.shape[0], k)`.
    """

    idx = kdtree_call(
        points=points, k=k, queries=queries,
        max_radius=max_radius,
        box_size=box_size,
        traversal_mode=traversal_mode,
        candidate_list=candidate_list
    )
    return idx
