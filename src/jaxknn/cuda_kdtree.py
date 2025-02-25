import numpy as np

from cudakdtree_jax_binding.cudakdtree_interface import kdtree_call
from cudakdtree_jax_binding.cudakdtree_interface import TraversalMode
from cudakdtree_jax_binding.cudakdtree_interface import CandidateList

# TODO: test where the best threshold is
HEAP_CANDIDATE_LIST_THRESHOLD = 10

def knn_cuda(points, k, queries=None, max_radius=np.inf, box_size=None, **kwargs):
    # TODO: default to cct in non-periodic case?
    traversal_mode = kwargs.get(
        "traversal_mode",
        TraversalMode.stack_free_bounds_tracking
    )
    candidate_list = kwargs.get(
        "candidate_list",
        CandidateList.fixed_list if k < HEAP_CANDIDATE_LIST_THRESHOLD else CandidateList.heap
    )

    idx = kdtree_call(
        points=points, k=k, queries=queries,
        max_radius=max_radius,
        box_size=box_size,
        traversal_mode=traversal_mode,
        candidate_list=candidate_list
    )
    return idx
