from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp

# Blind implementation


def F(l):
    # 2^l -1
    return (1 << l) - 1


def level(n):
    # (int)log_2(n + 1)
    return 31 - jax.lax.clz(n + 1)


def n_levels(N):
    return level(N - 1) + 1


def left_child_node(n):
    return 2*n + 1


def right_child_node(n):
    return 2*n + 2


def parent_node(n):
    return (n-1)//2


def subtree_size(s, L, N):
    l = level(s)
    first_lowest_level_child = ~((~s) << (L-l-1))
    inner_nodes = (1 << (L-l-1)) - 1
    lowest_level = jax.lax.min(
        jax.lax.max(0, N-first_lowest_level_child),
        1 << (L-l-1)
    )
    return inner_nodes + lowest_level


def segment_begin(s, L, N):
    l = level(s)
    number_of_left_siblings = s - ((1<<l) - 1)
    top_l_levels = (1<<l) - 1
    inner = number_of_left_siblings * ((1 << (L-l-1)) - 1)
    lowest = jax.lax.min(
        number_of_left_siblings * (1 << (L-l-1)),
        N - ((1 << (L-1)) -1)
    )
    return top_l_levels + inner + lowest


def split_dim(l, k):
    return l % k


def _update_tag(tag, idx, L, N):
    pivot_pos = segment_begin(tag, L, N) + subtree_size(left_child_node(tag), L, N)
    tag = jnp.select(
        [idx < pivot_pos, idx > pivot_pos],
        [left_child_node(tag), right_child_node(tag)],
        default=tag
    )
    return tag


def update_tag(tag, idx, l, L, N):
    def dont_update(tag, *args):
        return tag

    return jax.lax.cond(
        idx < F(l),  # Wastes time but probably better than changing array sizes
        dont_update,
        _update_tag,
        tag, idx, L, N
    )


def update_tags(tags, l, L, N):
    idx = jnp.arange(N)
    tags = jax.vmap(update_tag, in_axes=(0, 0, None, None, None))(tags, idx, l, L, N)
    return tags


def update_tags_python(tags, l, L, N):
    for i in range(N):
        if i < F(l):
            continue
        tags[i] = _update_tag(tags[i], i, L, N)
    return tags


def sort(tags, points, order, l, k):
    dim = split_dim(l, k)
    sort_idx = jnp.lexsort(keys=[points[:, dim], tags])
    tags = tags[sort_idx]
    points = points[sort_idx]

    return tags, points, order[sort_idx]


@jax.profiler.annotate_function
def build_tree(points, l_max):
    N, k = points.shape
    L = n_levels(N)

    tags = jnp.zeros(N, dtype=jnp.int32)
    order = jnp.arange(N)

    def sort_and_update_tags(state, xs):
        tags, points, order, l = state
        tags, points, order = sort(tags, points, order, l, k)

        tags = update_tags(tags, l, L, N)
        return (tags, points, order, l+1), 0

    (tags, points, order, l), _ = jax.lax.scan(
        f=sort_and_update_tags,
        init=(tags, points, order, 0),
        length=l_max
    )
    tags, points, order = sort(tags, points, order, l, k)
    return points, order


@jax.profiler.annotate_function
def knn_process_node(node, query, points, state, distance_sq_func):
    knn_distances_sq, knn_indices = state
    # Assumes every node is a point
    candidate_idx = node
    candidate_point = points[candidate_idx]

    # Support other distances/periodic boxes
    candidate_distance_sq = distance_sq_func(query, candidate_point)

    def update_candidate_list(candidate_distance_sq, candidate_idx, knn_distances_sq, knn_indices):
        # Last element gets replaced in any case
        knn_distances_sq = knn_distances_sq.at[-1].set(candidate_distance_sq)
        knn_indices = knn_indices.at[-1].set(candidate_idx)

        sort_idx = jnp.argsort(knn_distances_sq)
        return knn_distances_sq[sort_idx], knn_indices[sort_idx]

    state = jax.lax.cond(
        candidate_distance_sq < knn_distances_sq[-1],
        update_candidate_list,
        lambda candidate_distance_sq, candidate_idx, *state: state,
        candidate_distance_sq, candidate_idx, knn_distances_sq, knn_indices
    )
    return state


def knn_update_max_search_radius(state):
    knn_distances_sq, _ = state
    return jnp.sqrt(knn_distances_sq[-1])


def need_to_check_periodic_box(level, split_dim, box_size):
    n_dim = box_size.size
    return box_size[split_dim] > 0 and level < n_dim


def traverse_python(query, points, state,
                    process_node_func, update_max_search_radius_func,
                    box_size=None, distance_sq_func=None):
    N = points.shape[0]
    d = points.shape[1]

    if box_size is None:
        box_size = jnp.array([-1]*d)

    if distance_sq_func is None:
        distance_sq_func = lambda x, y: jnp.sum((x - y)**2)

    # start at root node
    current_node = 0

    # previous node, initialise to "parent" of root node
    previous_node = -1

    max_search_radius = jnp.inf

    while True:
        parent = parent_node(current_node) # check implementation against expression in paper
        if current_node >= N:
            # child does not exist, go back to parent
            previous_node = current_node
            current_node = parent
            continue

        from_parent = previous_node < current_node
        if from_parent:
            state = process_node_func(current_node, query, points, state, distance_sq_func)
            max_search_radius = update_max_search_radius_func(state)

        l = level(current_node)
        current_split_dim = split_dim(l, k=points.shape[1])
        split_pos = points[current_node, current_split_dim]
        query_pos = query[current_split_dim]
        signed_distance = query_pos - split_pos
        close_side = jnp.int32(signed_distance > 0)
        close_child_node = left_child_node(current_node) + close_side
        far_child_node = right_child_node(current_node) - close_side
        far_child_in_range = jnp.abs(signed_distance) <= max_search_radius

        if not far_child_in_range and box_size[current_split_dim] > 0:
            # Track far child far side bound
            far_child_in_range = (
                (query_pos < box_size[current_split_dim]/2
                 and query_pos < max_search_radius)
                or
                (query_pos >= box_size[current_split_dim]/2
                 and (box_size[current_split_dim] - query_pos) <= max_search_radius)
            )

        if from_parent:
            next_node = close_child_node
        elif previous_node == close_child_node:
            if far_child_in_range:
                next_node = far_child_node
            else:
                next_node = parent
        else:
            next_node = parent

        if next_node == -1:
            # Arrived back at the root, so we are done
            return state

        # next iteration
        previous_node = current_node
        current_node = next_node


def is_left_child(n):
    return bool(n%2)


def super_node(n, g):
    for _ in range(g):
        n = parent_node(n)
    return n


def expand_bounds_to_parent(bounds, current_node, parent, tree, world_bounds):
    d = tree.shape[1]

    l = level(parent)
    expand_dim = split_dim(l, k=d)
    expand_right = is_left_child(current_node)

    child_of_super_node = parent
    super_node_id = parent_node(child_of_super_node)
    super_node_dim = split_dim(level(super_node_id), k=d)
    while child_of_super_node > 0 and (expand_right != is_left_child(child_of_super_node) or super_node_dim != expand_dim):
        child_of_super_node = super_node_id
        super_node_id = parent_node(super_node_id)
        super_node_dim = split_dim(level(super_node_id), k=d)

    # # Assumes split dim = level mod d
    # child_of_super_node = super_node(current_node, d - 1)

    # # Move up to next super node with same split side and split dim
    # while child_of_super_node > 0 and expand_right != is_left_child(child_of_super_node):
    #     child_of_super_node = super_node(child_of_super_node, d)

    if super_node_id >= 0:
        expand_pos = tree[super_node_id, expand_dim]
    else:
        # bounds are world bounds
        if expand_right:
             # left child, expand upper bounds
            expand_pos = world_bounds[expand_dim, 1]
        else:
             # right child, expand lower bounds
            expand_pos = world_bounds[expand_dim, 0]

    if expand_right:
        # expand upper bounds
        bounds = bounds.at[expand_dim, 1].set(expand_pos)
    else:
        # expand lower bounds
        bounds = bounds.at[expand_dim, 0].set(expand_pos)

    return bounds


def shrink_bounds_to_child(bounds, current_node, child, tree):
    d = tree.shape[1]
    l = level(current_node)
    shrink_dim = split_dim(l, k=d)

    if is_left_child(child):
        # shrink upper bound
        bounds = bounds.at[shrink_dim, 1].set(tree[current_node, shrink_dim])
    else:
        bounds = bounds.at[shrink_dim, 0].set(tree[current_node, shrink_dim])

    return bounds


Stats = namedtuple("Stats", [
    "current_node",
    "is_left",
    "max_radius",
    "split_pos",
    "current_split_dim",
    "from_parent",
    "far_child_in_range",
    "next_node", "next_node_type",
    "bounds"
])

def traverse_stack_free_bounds_tracking_python(
    query, points, state,
    process_node_func, update_max_search_radius_func,
    box_size=None, distance_sq_func=None
):
    N = points.shape[0]
    d = points.shape[1]

    if box_size is None:
        box_size = jnp.array([-1]*d)

    if distance_sq_func is None:
        distance_sq_func = lambda x, y: jnp.sum((x - y)**2)

    # start at root node
    current_node = 0

    # previous node, initialise to "parent" of root node
    previous_node = -1

    max_search_radius = jnp.inf

    world_bounds = jnp.stack([jnp.zeros_like(box_size), box_size]).T
    bounds = world_bounds

    stats = []

    while True:
        parent = parent_node(current_node)
        if current_node >= N:
            # child does not exist, go back to parent
            previous_node = current_node
            current_node = parent
            continue

        from_parent = previous_node < current_node
        if from_parent:
            # Came from parent, update state
            state = process_node_func(current_node, query, points, state, distance_sq_func)
            max_search_radius = update_max_search_radius_func(state)

        l = level(current_node)
        current_split_dim = split_dim(l, k=points.shape[1])
        split_pos = points[current_node, current_split_dim]
        query_pos = query[current_split_dim]
        signed_distance = query_pos - split_pos
        close_side = jnp.int32(signed_distance > 0)
        close_child_node = left_child_node(current_node) + close_side

        # TODO: reorder logic to skip far side stuff if not necessary
        far_child_node = right_child_node(current_node) - close_side

        far_child_in_range = jnp.abs(signed_distance) <= max_search_radius

        # Check for periodic box and check if far child is within reach when wrapping
        if not far_child_in_range and box_size[current_split_dim] > 0:
            if is_left_child(far_child_node):
                # Check if lower bound is in reach
                d = jnp.abs(query_pos - bounds[current_split_dim, 0])
            else:
                # Check if lower bound is in reach
                d = jnp.abs(bounds[current_split_dim, 1] - query_pos)

            far_child_in_range = jnp.minimum(d, box_size[current_split_dim] - d) <= max_search_radius

        next_node = parent
        if from_parent:
            if close_child_node < N:
                next_node = close_child_node
            elif far_child_in_range and far_child_node < N:
                next_node = far_child_node
        elif previous_node == close_child_node:
            if far_child_in_range and far_child_node < N:
                next_node = far_child_node


        if next_node == -1:
            # Arrived back at the root, so we are done
            return state, stats

        # print("query: (%f, %f), node: %d, split: %f, split_dim: %d, from_parent: %d, far_child_in_range: %d, next: %d, current bounds: (%f, %f), (%f, %f)" % (
        #        *query,
        #        current_node,
        #        split_pos,
        #        current_split_dim,
        #        from_parent,
        #        far_child_in_range,
        #        next_node,
        #        *bounds[0],
        #        *bounds[1]))

        stats.append(Stats(
            current_node,
            is_left_child(current_node),
            max_search_radius,
            split_pos,
            current_split_dim,
            from_parent,
            far_child_in_range,
            next_node,
            "parent" if next_node == parent else ("close_child" if next_node == close_child_node else "far_child"),
            bounds))
        if next_node == parent:
            # assert False, "Check for being inside box"
            if jnp.all(jnp.abs(bounds - query[:, None]) > max_search_radius):
                # We finished the subtree and all current sides of the current bounds are out of reach, so no need to search other parts of the tree
                return state, stats
            bounds = expand_bounds_to_parent(bounds, current_node, next_node, tree=points, world_bounds=world_bounds)
        else:
            # assert False, "check if working outside of box, if so stop if box out of reach"
            if next_node < N:
                # Shrink, unless we go to a child that does not exist (next_node >= N)
                bounds = shrink_bounds_to_child(bounds, current_node, next_node, tree=points)

        # next iteration
        previous_node = current_node
        current_node = next_node


def traverse_stack_python(query, points, state,
                          process_node_func, update_max_search_radius_func,
                          box_size=None, distance_sq_func=None):
    N = points.shape[0]
    d = points.shape[1]

    assert box_size is not None

    if distance_sq_func is None:
        distance_sq_func = lambda x, y: jnp.sum((x - y)**2)

    current_node = 0

    max_search_radius = jnp.inf

    stack = []

    bounds = jnp.stack((jnp.zeros_like(box_size), box_size)).T

    while True:
        if current_node >= N:
            # Went to a child that does not exist, move back up the stack
            while True:
                if len(stack) == 0:
                    # Arrived back at top of the stack, so we are done
                    return state
                current_node, dist, current_split_dim, side, far_side_bound = stack.pop()
                split_pos = points[current_node, current_split_dim]
                bounds = bounds.at[current_split_dim, side].set(far_side_bound)
                bounds = bounds.at[current_split_dim, 1 - side].set(split_pos)
                if dist >= max_search_radius:
                    # Move up the stack
                    continue
                # Found a node within range, continue with traversal
                break

        # Process node
        state = process_node_func(current_node, query, points, state, distance_sq_func)
        max_search_radius = update_max_search_radius_func(state)

        l = level(current_node)
        current_split_dim = split_dim(l, k=d)

        split_pos = points[current_node, current_split_dim]
        query_pos = query[current_split_dim]
        signed_distance = query_pos - split_pos
        close_side = jnp.int32(signed_distance >= 0)
        close_child_node = left_child_node(current_node) + close_side
        far_child_node = right_child_node(current_node) - close_side

        far_child_dist = jnp.abs(signed_distance)

        far_child_far_bound = bounds[current_split_dim, 1 - close_side]
        if box_size[current_split_dim] > 0:
            far_child_far_bound_dist = jnp.abs(query_pos - far_child_far_bound)
            far_child_far_bound_dist = jnp.minimum(far_child_far_bound_dist, box_size[current_split_dim] - far_child_far_bound_dist)
            far_child_dist = jnp.minimum(far_child_far_bound_dist, far_child_dist)

        if far_child_node < N and far_child_dist < max_search_radius:
            stack.append((far_child_node, far_child_dist, current_split_dim, 1 - close_side, far_child_far_bound))

        bounds = bounds.at[current_split_dim, 1 - close_side].set(split_pos)
        current_node = close_child_node


Box = namedtuple("Box", ["lower", "upper"])


def project_box(box, point):
    return jax.lax.min(jax.lax.max(box.lower, point), box.upper)


def traverse_cct_stack_python(query, points, state, bounds, distance_func, process_node_func, update_max_search_radius_func):
    N = points.shape[0]
    # start at root node
    current_node = 0

    max_search_radius = jnp.inf

    stack = []

    closest_point_on_box = project_box(box=bounds, point=query)
    while True:
        if current_node >= N:
            # Went to a child that does not exist, move back up the stack
            while True:
                if len(stack) == 0:
                    # Arrived back at top of the stack, so we are done
                    return state
                closest_point_on_box, dist, current_node = stack.pop()
                if dist >= max_search_radius:
                    # Move up the stack
                    continue
                # Found a node within range, continue with traversal
                break

        # Process node
        state = process_node_func(current_node, query, points, state)
        max_search_radius = update_max_search_radius_func(state)

        l = level(current_node)
        current_split_dim = split_dim(l, k=points.shape[1])

        split_pos = points[current_node, current_split_dim]
        signed_distance = query[current_split_dim] - split_pos
        close_side = jnp.int32(signed_distance > 0)
        close_child_node = left_child_node(current_node) + close_side
        far_child_node = right_child_node(current_node) - close_side

        far_corner = closest_point_on_box
        far_corner = far_corner.at[current_split_dim].set(split_pos)
        far_corner_dist = distance_func(far_corner, query)
        if far_child_node < N and far_corner_dist < max_search_radius:
            stack.append((far_corner, far_corner_dist, far_child_node))

        current_node = close_child_node


@jax.profiler.annotate_function
def traverse(query, points, process_node_state, process_node_func, update_max_search_radius_func):
    N = points.shape[0]
    # start at root node
    current_node = 0

    # previous node, initialise to "parent" of root node
    previous_node = -1

    max_search_radius = jnp.inf

    @jax.profiler.annotate_function
    def traverse_body(state):
        previous_node, current_node, process_node_state, max_search_radius = state
        parent = parent_node(current_node) # check implementation against expression in paper

        def process_node(current_node, process_node_state, max_search_radius):
            process_node_state = process_node_func(current_node, query, points, process_node_state)
            max_search_radius = update_max_search_radius_func(process_node_state)
            return process_node_state, max_search_radius

        from_parent = previous_node < current_node

        process_node_state, max_search_radius = jax.lax.cond(
            from_parent,
            process_node,
            lambda _, process_node_state, max_search_radius: (process_node_state, max_search_radius),
            current_node, process_node_state, max_search_radius
        )

        l = level(current_node)
        current_split_dim = split_dim(l, k=points.shape[1])
        split_pos = points[current_node, current_split_dim]
        signed_distance = query[current_split_dim] - split_pos
        close_side = jnp.int32(signed_distance > 0)
        close_child_node = left_child_node(current_node) + close_side
        far_child_node = right_child_node(current_node) - close_side
        far_child_in_range = jnp.abs(signed_distance) <= max_search_radius

        next_node = jnp.select(
            [from_parent, previous_node == close_child_node],
            [close_child_node, jnp.select([far_child_in_range], [far_child_node], parent)],
            parent
        )

        previous_node, current_node = jax.lax.select(
            next_node >= N,
            jnp.array([next_node, current_node]),
            jnp.array([current_node, next_node])
        )

        return previous_node, current_node, process_node_state, max_search_radius

    def traverse_cond(state):
        previous_node, current_node, process_node_state, max_search_radius = state
        # If current_node == -1, we arrived back at the root, so we are done
        return current_node != -1

    previous_node, current_node, process_node_state, max_search_radius = jax.lax.while_loop(
        body_fun=traverse_body,
        cond_fun=traverse_cond,
        init_val=(previous_node, current_node, process_node_state, max_search_radius)
    )

    return process_node_state


def knn_traverse_vmap(points, k):
    knn_distances_sq = jnp.full(shape=(points.shape[0], k), fill_value=jnp.inf)
    knn_indices = jnp.full(shape=(points.shape[0], k), fill_value=-1, dtype=jnp.int32)

    # This is much slower (at least on CPU) than jax.lax.map. Possibly due to
    # the heterogenous computation while traversing. vmap+cond issues?
    knn_distances_sq, knn_indices = jax.vmap(traverse, in_axes=(0, None, 0, None, None))(
        points, points, (knn_distances_sq, knn_indices),
        knn_process_node, knn_update_max_search_radius
    )
    return knn_distances_sq, knn_indices


def knn_traverse_map(points, k):
    knn_distances_sq = jnp.full(shape=(points.shape[0], k), fill_value=jnp.inf)
    knn_indices = jnp.full(shape=(points.shape[0], k), fill_value=-1, dtype=jnp.int32)

    def traverse_wrapper(x):
        query, process_node_state = x
        return traverse(query, points, process_node_state, knn_process_node, knn_update_max_search_radius)

    knn_distances_sq, knn_indices = jax.lax.map(traverse_wrapper, (points, (knn_distances_sq, knn_indices)))
    return knn_distances_sq, knn_indices


def knn_traverse(points, queries, k, loop_mode,
                 max_radius=jnp.inf,
                 box_size=None,
                 distance_sq_func=None,
                 traverse_func=traverse,
                 knn_process_node_func=knn_process_node,
                 knn_update_max_search_radius_func=knn_update_max_search_radius):
    knn_distances_sq = jnp.full(shape=(queries.shape[0], k), fill_value=max_radius)
    knn_indices = jnp.full(shape=(queries.shape[0], k), fill_value=-1, dtype=jnp.int32)

    if loop_mode == "vmap":
        knn_distances_sq, knn_indices = jax.vmap(traverse_func, in_axes=(0, None, 0, None, None, None))(
            queries, points, (knn_distances_sq, knn_indices),
            knn_process_node_func, knn_update_max_search_radius_func, box_size, distance_sq_func
        )
    elif loop_mode == "map":
        def traverse_wrapper(x):
            query, process_node_state = x
            return traverse_func(query, points, process_node_state, knn_process_node_func, knn_update_max_search_radius_func, box_size, distance_sq_func)

        knn_distances_sq, knn_indices = jax.lax.map(traverse_wrapper, (queries, (knn_distances_sq, knn_indices)))
    elif loop_mode == "for":
        stats = []
        for i in range(queries.shape[0]):
            (d, idx), query_stats = traverse_func(queries[i], points, (knn_distances_sq[i], knn_indices[i]), knn_process_node_func, knn_update_max_search_radius_func, box_size, distance_sq_func)
            stats.append(query_stats)
            knn_distances_sq = knn_distances_sq.at[i].set(d)
            knn_indices = knn_indices.at[i].set(idx)
        return knn_distances_sq, knn_indices, stats
    else:
        raise ValueError(f"Loop mode {loop_mode} not supported.")

    return knn_distances_sq, knn_indices


@partial(jax.jit, static_argnames=["k", "l_max"])
def knn_jax(points, k, l_max):
    nodes, points = build_tree(points, l_max)
    return knn_traverse_map(points, k)


def print_state(tags, points):
    print("tags:" + "  ".join([f"{t:>3}" for t in tags]))
    print("x   :" + "  ".join([f"{p[0]:>3}" for p in points]))
    print("y   :" + "  ".join([f"{p[1]:>3}" for p in points]))
