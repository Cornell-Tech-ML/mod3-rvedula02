# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        # Check if this thread should process data
        if i < out_size:
            # Create thread-local index arrays
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            
            # Convert position to indices
            to_index(i, out_shape, out_index)
            # Calculate output position
            out_pos = index_to_position(out_index, out_strides)
            # Handle broadcasting for input
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            # Apply function and store result
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        # Check if this thread should process data
        if i < out_size:
            # Create thread-local index arrays
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            
            # Convert position to indices
            to_index(i, out_shape, out_index)
            # Calculate output position
            out_pos = index_to_position(out_index, out_strides)
            # Handle broadcasting for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            # Apply function and store result
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32  # This should match the test's expectation
    
    # Shared memory for partial sums within a block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    
    # Calculate thread and block index
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    
    # Global index for this thread
    global_id = bid * BLOCK_DIM + tid
    
    # Initialize shared memory
    if global_id < size:
        cache[tid] = a[global_id]
    else:
        cache[tid] = 0.0
        
    cuda.syncthreads()
    
    # Reduction in shared memory
    s = BLOCK_DIM // 2
    while s > 0:
        if tid < s and global_id < size:
            cache[tid] += cache[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write result for this block to global memory
    if tid == 0:
        out[bid] = cache[0]

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos >= out_size:
            return

        # Calculate position in output
        to_index(out_pos, out_shape, out_index)

        # Initialize reduction with first element
        cache[pos] = reduce_value

        # Pre-calculate input index array outside the loop
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        for i in range(len(out_shape)):
            a_index[i] = out_index[i]

        # Simplified loop without thread stride
        for k in range(a_shape[reduce_dim]):
            a_index[reduce_dim] = k
            in_pos = index_to_position(a_index, a_strides)
            cache[pos] = fn(cache[pos], a_storage[in_pos])

        cuda.syncthreads()

        # Fix parallel reduction within block
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2
        
        # Only first thread writes result
        if pos == 0:
            out[index_to_position(out_index, out_strides)] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Get thread indices
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    
    # Create shared memory for the block
    shared_a = cuda.shared.array((32, 32), numba.float64)
    shared_b = cuda.shared.array((32, 32), numba.float64)
    
    # Only process if within matrix bounds
    if i < size and j < size:
        # Load data into shared memory
        shared_a[i, j] = a[i * size + j]
        shared_b[i, j] = b[i * size + j]
    
    # Ensure all threads have loaded their data
    cuda.syncthreads()
    
    # Compute output element
    if i < size and j < size:
        # Initialize accumulator
        temp = 0.0
        # Perform dot product
        for k in range(size):
            temp += shared_a[i, k] * shared_b[k, j]
        # Write result to global memory
        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Get batch stride and index
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z

    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Get block indices
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Calculate global row and column
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Define tile size
    TILE_SIZE = 32

    # Allocate shared memory for tiles
    tile_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    tile_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Initialize accumulator
    acc = 0.0

    # Calculate number of tiles needed
    num_tiles = (a_shape[2] + TILE_SIZE - 1) // TILE_SIZE

    # Loop over tiles
    for t in range(num_tiles):
        # Load data into shared memory tiles with bounds checking

        if row < a_shape[1] and t * TILE_SIZE + tx < a_shape[2]:
            a_pos = (
                batch * a_batch_stride
                + row * a_strides[1]
                + (t * TILE_SIZE + tx) * a_strides[2]
            )
            tile_a[ty, tx] = a_storage[a_pos]
        else:
            tile_a[ty, tx] = 0.0

        if t * TILE_SIZE + ty < b_shape[1] and col < b_shape[2]:
            b_pos = (
                batch * b_batch_stride
                + (t * TILE_SIZE + ty) * b_strides[1]
                + col * b_strides[2]
            )
            tile_b[ty, tx] = b_storage[b_pos]
        else:
            tile_b[ty, tx] = 0.0

        # Synchronize threads
        cuda.syncthreads()

        # Compute partial dot product for this tile
        if row < out_shape[1] and col < out_shape[2]:
            for k in range(min(TILE_SIZE, a_shape[2] - t * TILE_SIZE)):
                acc += tile_a[ty, k] * tile_b[k, tx]

        # Synchronize before loading next tile
        cuda.syncthreads()

    # Write final result to global memory
    if row < out_shape[1] and col < out_shape[2]:
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out[out_pos] = acc
            

tensor_matrix_multiply = jit(_tensor_matrix_multiply)
