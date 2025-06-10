import pytest

import numpy as np
import jax.numpy as jnp

from jaxknn.onedee import knn_1d
from jaxknn.scipy_kdtree import knn_scipy


def test_knn_1d_random():
    # Test with random uniform points
    n_p = 100
    n_q = 10
    k = 10
    
    x = np.random.uniform(size=n_p)
    q = np.random.uniform(size=n_q)
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)


def test_knn_1d_sequential():
    # Test with sequential points
    x = np.arange(10)
    q = np.array([0, 4.5, 9])
    k = 3
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)


def test_knn_1d_periodic():
    # Test periodic boundary conditions
    x = np.arange(10)
    q = np.array([0.1, 4.6, 9.4])
    k = 3
    box_size = 10.0
    
    idx_1d = knn_1d(
        points=jnp.asarray(x), 
        queries=jnp.asarray(q), 
        k=k,
        box_size=box_size
    )
    idx_scipy = knn_scipy(
        points=x[:, None],
        queries=q[:, None], 
        k=k,
        box_size=box_size
    )
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)
    
    # For point 0, nearest neighbors should include points from end of array
    assert 9 in idx_1d[0]
    # For point 9, nearest neighbors should include points from start of array  
    assert 0 in idx_1d[2]


def test_knn_1d_edge_cases():
    # Test k=number of points
    x = np.arange(5)
    q = np.array([2.6])
    k = 5
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k, bruteforce=True)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)

    np.testing.assert_array_equal(idx_1d, idx_scipy)

    # When using the sorting variant, 2*k + 1 must be less than or equal to n_points
    x = np.arange(11)
    q = np.array([2.6])
    k = 5
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)

    np.testing.assert_array_equal(idx_1d, idx_scipy)


def test_knn_1d_random_periodic():
    # Test periodic boundary conditions with random data
    n_p = 100
    k = 10
    box_size = 1.0
    
    x = np.random.uniform(size=n_p)
    q = np.array([0.0, 0.1, 0.95]) # Points near boundaries
    
    idx_1d = knn_1d(
        points=jnp.asarray(x),
        queries=jnp.asarray(q),
        k=k,
        box_size=box_size
    )
    
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k, box_size=box_size)
    np.testing.assert_array_equal(idx_1d, idx_scipy)


def test_knn_1d_invalid_inputs():
    x = np.random.uniform(size=10)
    q = np.random.uniform(size=5)
    
    # Test invalid k
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=0)
        
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=-1)
        
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=11) # k > n_points

    # Test invalid box_size
    # Box size must be positive if provided
    # Test with negative box_size
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=3, box_size=-1.0)
    # Test with zero box_size
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=3, box_size=0.0)


def test_knn_1d_empty_inputs():
    # Test empty points array
    x = np.array([])
    q = np.array([1.0])
    
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=1)
        
    # Test empty queries array
    x = np.array([1.0])
    q = np.array([])
    
    with pytest.raises(ValueError):
        knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=1)


def test_knn_1d_brute_force():
    # Test small dataset where brute force is suitable
    x = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
    q = np.array([0.3, 0.6, 0.8])
    k = 2
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k, bruteforce=True)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)


def test_knn_1d_periodic_bruteforce():
    # Test periodic boundary conditions with bruteforce
    x = np.array([0.1, 0.2, 0.8, 0.9])
    q = np.array([0.02, 0.96]) # Points very close to boundaries
    k = 3
    box_size = 1.0
    
    idx_1d = knn_1d(
        points=jnp.asarray(x),
        queries=jnp.asarray(q),
        k=k,
        box_size=box_size,
        bruteforce=True
    )
    
    idx_scipy = knn_scipy(
        points=x[:, None],
        queries=q[:, None], 
        k=k,
        box_size=box_size
    )
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)
    
    # Check that nearest neighbors wrap around boundary
    assert idx_1d[0, 1] == 3  # 0.9 should be near 0.02
    assert idx_1d[1, 1] == 0  # 0.1 should be near 0.96


def test_knn_1d_single_query():
    # Test with single query point
    n_p = 50
    x = np.random.uniform(size=n_p)
    q = np.array([0.5])
    k = 5
    
    idx_1d = knn_1d(points=jnp.asarray(x), queries=jnp.asarray(q), k=k)
    idx_scipy = knn_scipy(points=x[:, None], queries=q[:, None], k=k)
    
    np.testing.assert_array_equal(idx_1d, idx_scipy)
