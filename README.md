# k-nearest neighbours for JAX

This repo implements k-nearest neighbour searches based on kd-trees in JAX. 
It relies on a couple different implementations:

- CPU: Wrapping `scipy.spatial.KDTree` using `jax.pure_callback`.
- GPU: Wrapping [cudaKDTree](https://github.com/ingowald/cudaKDTree) using the JAX FFI: [cudakdtree_jax_binding](https://github.com/tilmantroester/cudakdtree_jax_binding).
- WIP pure JAX implementation of the kdtree algorithms (super slow but useful for testing algorithms).

## Installation

The GPU code relies on [cudakdtree_jax_binding](https://github.com/tilmantroester/). This is installed when using the `[gpu]` dependency:

```
pip install git+https://github.com/tilmantroester/jaxknn.git#egg=jaxknn[gpu]
```
Or `pip install .[gpu]`, if installing from a local copy of the repository. 
Alternatively, `cudakdtree_jax_binding` can also be installed directly: 
```
pip install git+https://github.com/tilmantroester/cudakdtree_jax_binding
```

## Usage

```python
import jax
import jax.numpy as jnp

from jaxknn.scipy_kdtree import knn_scipy
from jaxknn.cuda_kdtree import knn_cuda

points = jax.random.uniform(key=jax.random.key(42), shape=(16**3, 3))
box_size = (1.0, 1.0, 1.0)  # Toroidal topology
k = 16

scipy_idx = knn_scipy(
    points=points, k=k, box_size=box_size
)

cuda_idx = knn_cuda(
    points=points, k=k, box_size=box_size,
)
```