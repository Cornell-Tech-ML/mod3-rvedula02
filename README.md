# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/content/mod3-rvedula02/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-rvedula02/minitorch/fast_ops.py (164) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
        # Check if tensors are stride-aligned                            | 
        # Check for aligned tensors                                      | 
        is_aligned = (                                                   | 
            len(out_strides) == len(in_strides)                          | 
            and np.array_equal(out_strides, in_strides)                  | 
            and np.array_equal(out_shape, in_shape)                      | 
        )                                                                | 
                                                                         | 
        if is_aligned:                                                   | 
            # Fast path for aligned tensors                              | 
            for idx in prange(len(out)):---------------------------------| #0
                out[idx] = fn(in_storage[idx])                           | 
            return                                                       | 
                                                                         | 
        # Calculate total elements in output                             | 
        total_elements = np.prod(out_shape)------------------------------| #1
                                                                         | 
        # Main parallel processing loop                                  | 
        for i in prange(total_elements):---------------------------------| #2
            # Create index buffers per thread                            | 
            out_index = np.empty(len(out_shape), np.int32)               | 
            in_index = np.empty(len(in_shape), np.int32)                 | 
            # Convert position to indices                                | 
            to_index(i, out_shape, out_index)                            | 
            # Calculate output position                                  | 
            o_pos = index_to_position(out_index, out_strides)            | 
            # Map output index to input index                            | 
            broadcast_index(out_index, out_shape, in_shape, in_index)    | 
            # Calculate input position                                   | 
            i_pos = index_to_position(in_index, in_strides)              | 
            # Apply function                                             | 
            out[o_pos] = fn(in_storage[i_pos])                           | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (193) is hoisted out of the 
parallel loop labelled #2 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (194) is hoisted out of the 
parallel loop labelled #2 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/content/mod3-rvedula02/minitorch/fast_ops.py (232)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-rvedula02/minitorch/fast_ops.py (232) 
----------------------------------------------------------------------------|loop #ID
    def _zip(                                                               | 
        out: Storage,                                                       | 
        out_shape: Shape,                                                   | 
        out_strides: Strides,                                               | 
        a_storage: Storage,                                                 | 
        a_shape: Shape,                                                     | 
        a_strides: Strides,                                                 | 
        b_storage: Storage,                                                 | 
        b_shape: Shape,                                                     | 
        b_strides: Strides,                                                 | 
    ) -> None:                                                              | 
        # TODO: Implement for Task 3.1.                                     | 
        # Special case - when tensors are stride-aligned, avoid indexing    | 
        # Check if tensors are stride-aligned                               | 
        if (                                                                | 
            len(out_strides) == len(a_strides) == len(b_strides)            | 
            and np.array_equal(out_strides, a_strides)                      | 
            and np.array_equal(out_strides, b_strides)                      | 
            and np.array_equal(out_shape, a_shape)                          | 
            and np.array_equal(out_shape, b_shape)                          | 
        ):                                                                  | 
            # Optimized path for stride-aligned tensors                     | 
            for idx in prange(len(out)):------------------------------------| #3
                out[idx] = fn(a_storage[idx], b_storage[idx])               | 
            return                                                          | 
                                                                            | 
        # Handle tensors with non-aligned strides                           | 
        total_elements = 1                                                  | 
        for dim_size in out_shape:                                          | 
            total_elements *= dim_size                                      | 
                                                                            | 
        for i in prange(total_elements):------------------------------------| #4
            # Create index buffers per thread                               | 
            out_index = np.empty(len(out_shape), np.int32)                  | 
            a_index = np.empty(len(out_shape), np.int32)                    | 
            b_index = np.empty(len(out_shape), np.int32)                    | 
            # Convert position to indices and calculate positions           | 
            to_index(i, out_shape, out_index)                               | 
            o_pos = index_to_position(out_index, out_strides)               | 
            broadcast_index(out_index, out_shape, a_shape, a_index)         | 
            a_pos = index_to_position(a_index, a_strides)                   | 
            broadcast_index(out_index, out_shape, b_shape, b_index)         | 
            b_pos = index_to_position(b_index, b_strides)                   | 
            out[o_pos] = fn(a_storage[a_pos], b_storage[b_pos])             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #3, #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (265) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (266) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (267) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/content/mod3-rvedula02/minitorch/fast_ops.py (301)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-rvedula02/minitorch/fast_ops.py (301) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # Calculate output size                                    | 
        size = len(out_shape)                                      | 
        out_size = 1                                               | 
        for i in range(size):                                      | 
            out_size *= out_shape[i]                               | 
                                                                   | 
        # Main parallel loop over output positions                 | 
        for i in prange(out_size):---------------------------------| #5
            # Create thread-local index buffers                    | 
            out_index = np.empty(size, np.int32)                   | 
            a_index = np.empty(size, np.int32)                     | 
                                                                   | 
            # Convert position to indices                          | 
            to_index(i, out_shape, out_index)                      | 
                                                                   | 
            # Calculate output position                            | 
            o_pos = index_to_position(out_index, out_strides)      | 
                                                                   | 
            # Copy output index to a_index                         | 
            for j in range(size):                                  | 
                a_index[j] = out_index[j]                          | 
                                                                   | 
            # Initialize reduction with first element              | 
            a_index[reduce_dim] = 0                                | 
            pos = index_to_position(a_index, a_strides)            | 
            reduced = a_storage[pos]                               | 
                                                                   | 
            # Inner reduction loop starting from second element    | 
            for j in range(1, a_shape[reduce_dim]):                | 
                a_index[reduce_dim] = j                            | 
                pos = index_to_position(a_index, a_strides)        | 
                # Apply reduction function                         | 
                reduced = fn(reduced, a_storage[pos])              | 
                                                                   | 
            # Store result                                         | 
            out[o_pos] = reduced                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (319) is hoisted out of the 
parallel loop labelled #5 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(size, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-rvedula02/minitorch/fast_ops.py (320) is hoisted out of the 
parallel loop labelled #5 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(size, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/content/mod3-rvedula02/minitorch/fast_ops.py (350)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-rvedula02/minitorch/fast_ops.py (350) 
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              | 
    out: Storage,                                                                         | 
    out_shape: Shape,                                                                     | 
    out_strides: Strides,                                                                 | 
    a_storage: Storage,                                                                   | 
    a_shape: Shape,                                                                       | 
    a_strides: Strides,                                                                   | 
    b_storage: Storage,                                                                   | 
    b_shape: Shape,                                                                       | 
    b_strides: Strides,                                                                   | 
) -> None:                                                                                | 
    """NUMBA tensor matrix multiply function.                                             | 
                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                           | 
                                                                                          | 
    ```                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    ```                                                                                   | 
                                                                                          | 
    Optimizations:                                                                        | 
                                                                                          | 
    * Outer loop in parallel                                                              | 
    * No index buffers or function calls                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                | 
                                                                                          | 
                                                                                          | 
    Args:                                                                                 | 
    ----                                                                                  | 
        out (Storage): storage for `out` tensor                                           | 
        out_shape (Shape): shape for `out` tensor                                         | 
        out_strides (Strides): strides for `out` tensor                                   | 
        a_storage (Storage): storage for `a` tensor                                       | 
        a_shape (Shape): shape for `a` tensor                                             | 
        a_strides (Strides): strides for `a` tensor                                       | 
        b_storage (Storage): storage for `b` tensor                                       | 
        b_shape (Shape): shape for `b` tensor                                             | 
        b_strides (Strides): strides for `b` tensor                                       | 
                                                                                          | 
    Returns:                                                                              | 
    -------                                                                               | 
        None : Fills in `out`                                                             | 
                                                                                          | 
    """                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                | 
                                                                                          | 
    # TODO: Implement for Task 3.2.                                                       | 
    for i in prange(out_shape[0]):--------------------------------------------------------| #6
        for j in range(out_shape[1]):                                                     | 
            for k in range(out_shape[2]):                                                 | 
                # Initialize accumulator for dot product                                  | 
                acc = 0.0                                                                 | 
                # Compute dot product along shared dimension                              | 
                for l in range(a_shape[-1]):                                              | 
                    # Get positions in a and b storage                                    | 
                    a_pos = i * a_batch_stride + j * a_strides[1] + l * a_strides[2]      | 
                    b_pos = i * b_batch_stride + l * b_strides[1] + k * b_strides[2]      | 
                    # Multiply and accumulate                                             | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                            | 
                # Write result to output                                                  | 
                out_pos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    | 
                out[out_pos] = acc                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  0  loss  7.832233401518151 correct 33 avg time per epoch: 5.2149s
Epoch  10  loss  5.412717761184166 correct 41 avg time per epoch: 1.4964s
Epoch  20  loss  5.762394450988035 correct 41 avg time per epoch: 1.4908s
Epoch  30  loss  3.2938839005145324 correct 43 avg time per epoch: 1.5367s
Epoch  40  loss  3.3698140730100365 correct 45 avg time per epoch: 1.4624s
Epoch  50  loss  3.668557447112881 correct 46 avg time per epoch: 1.4672s
Epoch  60  loss  4.019975108563675 correct 46 avg time per epoch: 1.4593s
Epoch  70  loss  1.7815207081099298 correct 43 avg time per epoch: 1.4820s
Epoch  80  loss  8.328270042589349 correct 42 avg time per epoch: 1.5535s
Epoch  90  loss  2.014697173102404 correct 48 avg time per epoch: 1.4607s
Epoch  100  loss  0.8937030342892227 correct 47 avg time per epoch: 1.4650s
Epoch  110  loss  0.9864958347736096 correct 49 avg time per epoch: 1.4661s
Epoch  120  loss  1.078975715144775 correct 49 avg time per epoch: 1.5076s
Epoch  130  loss  1.6160013515848677 correct 47 avg time per epoch: 1.5136s
Epoch  140  loss  0.9074784173386481 correct 49 avg time per epoch: 1.4660s
Epoch  150  loss  0.6575310518289853 correct 48 avg time per epoch: 1.4677s
Epoch  160  loss  0.9063791919041951 correct 47 avg time per epoch: 1.4619s
Epoch  170  loss  0.42676874940854603 correct 50 avg time per epoch: 1.5090s
Epoch  180  loss  0.37510490927146845 correct 47 avg time per epoch: 1.4940s
Epoch  190  loss  0.43853305208533033 correct 49 avg time per epoch: 1.4658s
Epoch  200  loss  1.9458506384159224 correct 50 avg time per epoch: 1.4594s
Epoch  210  loss  0.601552122086978 correct 49 avg time per epoch: 1.4649s
Epoch  220  loss  1.151138513921821 correct 49 avg time per epoch: 1.5279s
Epoch  230  loss  0.062770711827069 correct 49 avg time per epoch: 1.4894s
Epoch  240  loss  0.8313261270318788 correct 49 avg time per epoch: 1.4604s
Epoch  250  loss  0.24691403155108235 correct 49 avg time per epoch: 1.4652s
Epoch  260  loss  0.9493993245481509 correct 49 avg time per epoch: 1.4510s
Epoch  270  loss  0.3826160966239492 correct 49 avg time per epoch: 1.5100s
Epoch  280  loss  1.2167210579017953 correct 49 avg time per epoch: 1.4892s
Epoch  290  loss  0.7071436638318349 correct 49 avg time per epoch: 1.4683s
Epoch  300  loss  0.60096332769554 correct 49 avg time per epoch: 1.4587s
Epoch  310  loss  1.465141144586592 correct 50 avg time per epoch: 1.4625s
Epoch  320  loss  1.8350472328630123 correct 50 avg time per epoch: 1.5242s
Epoch  330  loss  0.677959138050272 correct 49 avg time per epoch: 1.4875s
Epoch  340  loss  0.18358329631055156 correct 49 avg time per epoch: 1.4569s
Epoch  350  loss  0.4386438500129852 correct 49 avg time per epoch: 1.4518s
Epoch  360  loss  1.8069253438173556 correct 49 avg time per epoch: 1.4668s
Epoch  370  loss  2.1449322721117485 correct 47 avg time per epoch: 1.5286s
Epoch  380  loss  0.376340193999036 correct 50 avg time per epoch: 1.4738s
Epoch  390  loss  0.5297671231205716 correct 49 avg time per epoch: 1.4579s
Epoch  400  loss  0.9670851514262506 correct 47 avg time per epoch: 1.4625s
Epoch  410  loss  0.6303386174720015 correct 49 avg time per epoch: 1.4643s
Epoch  420  loss  1.147844600548242 correct 49 avg time per epoch: 1.5331s
Epoch  430  loss  2.0524833916300733 correct 45 avg time per epoch: 1.4673s
Epoch  440  loss  1.3387592766296452 correct 50 avg time per epoch: 1.4650s
Epoch  450  loss  0.22705329197542173 correct 49 avg time per epoch: 1.4648s
Epoch  460  loss  1.3899395673552268 correct 47 avg time per epoch: 1.4637s
Epoch  470  loss  1.9058412584886035 correct 49 avg time per epoch: 1.5404s
Epoch  480  loss  1.4766336228398806 correct 48 avg time per epoch: 1.4646s
Epoch  490  loss  0.1817903772080455 correct 49 avg time per epoch: 1.4625s



!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  0  loss  7.552710431011444 correct 29 avg time per epoch: 5.0753s
Epoch  10  loss  6.198419819647562 correct 31 avg time per epoch: 1.4110s
Epoch  20  loss  4.111209277931213 correct 31 avg time per epoch: 1.3849s
Epoch  30  loss  4.812343019406235 correct 43 avg time per epoch: 1.3906s
Epoch  40  loss  5.840412794849038 correct 43 avg time per epoch: 1.3850s
Epoch  50  loss  8.255191616573923 correct 43 avg time per epoch: 1.4059s
Epoch  60  loss  4.054836628299103 correct 44 avg time per epoch: 1.4322s
Epoch  70  loss  3.2120419786109453 correct 49 avg time per epoch: 1.4114s
Epoch  80  loss  4.326843679206335 correct 44 avg time per epoch: 1.3754s
Epoch  90  loss  2.658546485194626 correct 49 avg time per epoch: 1.3818s
Epoch  100  loss  2.1336953977882724 correct 50 avg time per epoch: 1.3831s
Epoch  110  loss  2.3176932494320366 correct 48 avg time per epoch: 1.3847s
Epoch  120  loss  1.6624089400927944 correct 49 avg time per epoch: 1.3712s
Epoch  130  loss  0.9140010433710615 correct 49 avg time per epoch: 1.4140s
Epoch  140  loss  1.1771351917300916 correct 48 avg time per epoch: 1.4197s
Epoch  150  loss  1.1549033708993606 correct 48 avg time per epoch: 1.3736s
Epoch  160  loss  1.577992716383179 correct 47 avg time per epoch: 1.3877s
Epoch  170  loss  1.9324687114304495 correct 50 avg time per epoch: 1.3871s
Epoch  180  loss  2.6649672755924465 correct 46 avg time per epoch: 1.3758s
Epoch  190  loss  0.42762785982536194 correct 50 avg time per epoch: 1.3770s
Epoch  200  loss  1.9651761941188057 correct 49 avg time per epoch: 1.4015s
Epoch  210  loss  0.6899375922909277 correct 48 avg time per epoch: 1.4339s
Epoch  220  loss  1.4055937206891287 correct 50 avg time per epoch: 1.3866s
Epoch  230  loss  0.9963666830457736 correct 50 avg time per epoch: 1.3823s
Epoch  240  loss  0.5242258980797265 correct 47 avg time per epoch: 1.3950s
Epoch  250  loss  0.9781577218011668 correct 49 avg time per epoch: 1.4494s
Epoch  260  loss  1.2261955625032244 correct 50 avg time per epoch: 1.3895s
Epoch  270  loss  1.0634030069826295 correct 50 avg time per epoch: 1.4488s
Epoch  280  loss  1.4304713724289142 correct 47 avg time per epoch: 1.4188s
Epoch  290  loss  0.18607064585978006 correct 50 avg time per epoch: 1.3948s
Epoch  300  loss  0.8723986736400485 correct 50 avg time per epoch: 1.4002s
Epoch  310  loss  0.27642661291359755 correct 50 avg time per epoch: 1.3966s
Epoch  320  loss  0.8414349284591657 correct 50 avg time per epoch: 1.4024s
Epoch  330  loss  0.6951095883083185 correct 50 avg time per epoch: 1.4231s
Epoch  340  loss  1.1433943034584026 correct 50 avg time per epoch: 1.4719s
Epoch  350  loss  0.41466889022388853 correct 50 avg time per epoch: 1.3949s
Epoch  360  loss  0.13071118308162602 correct 49 avg time per epoch: 1.3995s
Epoch  370  loss  0.4387695183009401 correct 50 avg time per epoch: 1.3998s
Epoch  380  loss  1.0797074416194974 correct 50 avg time per epoch: 1.3979s
Epoch  390  loss  0.9763763281666527 correct 50 avg time per epoch: 1.4082s
Epoch  400  loss  1.0590790972504718 correct 50 avg time per epoch: 1.4498s
Epoch  410  loss  1.0552920036548588 correct 50 avg time per epoch: 1.4176s
Epoch  420  loss  0.5382703249799521 correct 50 avg time per epoch: 1.3850s
Epoch  430  loss  1.0683844424404072 correct 50 avg time per epoch: 1.3891s
Epoch  440  loss  0.4123695578887477 correct 50 avg time per epoch: 1.3781s
Epoch  450  loss  0.12438052868491001 correct 50 avg time per epoch: 1.3970s
Epoch  460  loss  0.7963112189899139 correct 50 avg time per epoch: 1.3889s
Epoch  470  loss  0.48697820961072963 correct 50 avg time per epoch: 1.4719s
Epoch  480  loss  0.4608768739635145 correct 50 avg time per epoch: 1.3896s
Epoch  490  loss  0.39260686958742164 correct 50 avg time per epoch: 1.3982s



!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  0  loss  4.833392354418886 correct 36 avg time per epoch: 5.0070s
Epoch  10  loss  2.986064245691171 correct 45 avg time per epoch: 1.3801s
Epoch  20  loss  3.500726374830429 correct 44 avg time per epoch: 1.3808s
Epoch  30  loss  2.1373373083953338 correct 45 avg time per epoch: 1.3800s
Epoch  40  loss  1.693534623569208 correct 45 avg time per epoch: 1.3759s
Epoch  50  loss  1.8338256394697785 correct 45 avg time per epoch: 1.3740s
Epoch  60  loss  1.548690340307398 correct 45 avg time per epoch: 1.3713s
Epoch  70  loss  2.8033296065122206 correct 45 avg time per epoch: 1.4216s
Epoch  80  loss  4.1431384453789715 correct 47 avg time per epoch: 1.4008s
Epoch  90  loss  2.5868790090615814 correct 47 avg time per epoch: 1.3681s
Epoch  100  loss  1.582925419955473 correct 47 avg time per epoch: 1.3747s
Epoch  110  loss  4.0767665288092125 correct 47 avg time per epoch: 1.3680s
Epoch  120  loss  2.151972354485356 correct 47 avg time per epoch: 1.3710s
Epoch  130  loss  1.3506352240962052 correct 45 avg time per epoch: 1.3636s
Epoch  140  loss  1.9970302777132216 correct 48 avg time per epoch: 1.3840s
Epoch  150  loss  1.0848851361095877 correct 47 avg time per epoch: 1.4513s
Epoch  160  loss  2.6333833216915408 correct 48 avg time per epoch: 1.4527s
Epoch  170  loss  1.1263876900085679 correct 48 avg time per epoch: 1.3753s
Epoch  180  loss  3.7844971537916607 correct 49 avg time per epoch: 1.3606s
Epoch  190  loss  2.488257426734527 correct 49 avg time per epoch: 1.3689s
Epoch  200  loss  1.3068163415159 correct 49 avg time per epoch: 1.3734s
Epoch  210  loss  0.9374744274214563 correct 49 avg time per epoch: 1.4147s
Epoch  220  loss  1.6677815029801617 correct 49 avg time per epoch: 1.4201s
Epoch  230  loss  1.7055999238440265 correct 49 avg time per epoch: 1.3694s
Epoch  240  loss  0.8152699977379705 correct 49 avg time per epoch: 1.3759s
Epoch  250  loss  0.9412724081630778 correct 49 avg time per epoch: 1.3763s
Epoch  260  loss  1.5545128086295048 correct 49 avg time per epoch: 1.3734s
Epoch  270  loss  0.6996457204548723 correct 49 avg time per epoch: 1.3634s
Epoch  280  loss  0.6617827737803551 correct 50 avg time per epoch: 1.3701s
Epoch  290  loss  0.8741184219420508 correct 50 avg time per epoch: 1.4385s
Epoch  300  loss  0.42396845530481053 correct 50 avg time per epoch: 1.4281s
Epoch  310  loss  0.3551385514626223 correct 50 avg time per epoch: 1.3708s
Epoch  320  loss  0.5863251514203192 correct 50 avg time per epoch: 1.3860s
Epoch  330  loss  0.47405882366282753 correct 50 avg time per epoch: 1.3819s
Epoch  340  loss  0.19717107222003485 correct 50 avg time per epoch: 1.3725s
Epoch  350  loss  0.5291737393446462 correct 50 avg time per epoch: 1.3704s
Epoch  360  loss  0.4419000555322468 correct 50 avg time per epoch: 1.4132s
Epoch  370  loss  0.10529739912904781 correct 50 avg time per epoch: 1.4034s
Epoch  380  loss  0.6234508019776529 correct 50 avg time per epoch: 1.3776s
Epoch  390  loss  0.11984744573782417 correct 50 avg time per epoch: 1.3872s
Epoch  400  loss  0.32919104189124354 correct 50 avg time per epoch: 1.3781s
Epoch  410  loss  0.30117015387516854 correct 50 avg time per epoch: 1.3866s
Epoch  420  loss  0.2505441129753058 correct 50 avg time per epoch: 1.3765s
Epoch  430  loss  0.266044735737926 correct 50 avg time per epoch: 1.4750s
Epoch  440  loss  0.4666840721819492 correct 50 avg time per epoch: 1.4492s
Epoch  450  loss  0.07359215104748186 correct 50 avg time per epoch: 1.3841s
Epoch  460  loss  0.12011737017756006 correct 50 avg time per epoch: 1.3815s
Epoch  470  loss  0.41890415754702465 correct 50 avg time per epoch: 1.3781s
Epoch  480  loss  0.16926741672980053 correct 50 avg time per epoch: 1.3814s
Epoch  490  loss  0.08343659521105198 correct 50 avg time per epoch: 1.3795s



!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

 Epoch  0  loss  10.805244505263492 correct 22 avg time per epoch: 20.2660s
Epoch  10  loss  3.1290339126910043 correct 47 avg time per epoch: 0.1254s
Epoch  20  loss  2.0168910519129524 correct 47 avg time per epoch: 0.1573s
Epoch  30  loss  1.8566643213199403 correct 47 avg time per epoch: 0.0871s
Epoch  40  loss  1.2954816388089332 correct 50 avg time per epoch: 0.0970s
Epoch  50  loss  0.45997646132297015 correct 47 avg time per epoch: 0.0846s
Epoch  60  loss  0.0631777137855668 correct 50 avg time per epoch: 0.0862s
Epoch  70  loss  0.049804933114737666 correct 47 avg time per epoch: 0.0867s
Epoch  80  loss  1.4423318804885141 correct 50 avg time per epoch: 0.0868s
Epoch  90  loss  0.3586764788341006 correct 47 avg time per epoch: 0.0868s
Epoch  100  loss  1.6110176527725253 correct 48 avg time per epoch: 0.0866s
Epoch  110  loss  0.9115367443652647 correct 50 avg time per epoch: 0.0845s
Epoch  120  loss  2.2119425550492267 correct 47 avg time per epoch: 0.0860s
Epoch  130  loss  0.045965198088243606 correct 47 avg time per epoch: 0.0855s
Epoch  140  loss  1.7652296627461963 correct 47 avg time per epoch: 0.1474s
Epoch  150  loss  0.46063726570522046 correct 47 avg time per epoch: 0.1413s
Epoch  160  loss  0.7257906431198349 correct 47 avg time per epoch: 0.0873s
Epoch  170  loss  0.49543159884381566 correct 48 avg time per epoch: 0.0864s
Epoch  180  loss  1.2868325025475649 correct 47 avg time per epoch: 0.0858s
Epoch  190  loss  0.9174522441214121 correct 50 avg time per epoch: 0.0879s
Epoch  200  loss  0.5427998337916935 correct 50 avg time per epoch: 0.0870s
Epoch  210  loss  0.33999402914800775 correct 49 avg time per epoch: 0.0898s
Epoch  220  loss  0.5282480478787853 correct 49 avg time per epoch: 0.0876s
Epoch  230  loss  0.9693052495605654 correct 49 avg time per epoch: 0.0872s
Epoch  240  loss  1.724407173032369 correct 48 avg time per epoch: 0.0881s
Epoch  250  loss  0.34876621337042013 correct 49 avg time per epoch: 0.0855s
Epoch  260  loss  0.0764512469380825 correct 48 avg time per epoch: 0.0936s
Epoch  270  loss  0.031973875774105195 correct 49 avg time per epoch: 0.1368s
Epoch  280  loss  0.05941404372840497 correct 50 avg time per epoch: 0.1399s
Epoch  290  loss  0.0002681485830882684 correct 48 avg time per epoch: 0.0876s
Epoch  300  loss  1.3590750006953187 correct 50 avg time per epoch: 0.0863s
Epoch  310  loss  1.2169425224147794 correct 48 avg time per epoch: 0.0891s
Epoch  320  loss  0.8184476493699158 correct 48 avg time per epoch: 0.0857s
Epoch  330  loss  1.2858710457553415 correct 48 avg time per epoch: 0.0861s
Epoch  340  loss  1.1997736165332884 correct 50 avg time per epoch: 0.0871s
Epoch  350  loss  0.07634647288826245 correct 48 avg time per epoch: 0.0866s
Epoch  360  loss  1.191847023465342 correct 48 avg time per epoch: 0.0862s
Epoch  370  loss  0.2224658309496943 correct 49 avg time per epoch: 0.0853s
Epoch  380  loss  0.10227298426210968 correct 49 avg time per epoch: 0.0848s
Epoch  390  loss  1.370739839723675 correct 48 avg time per epoch: 0.0853s
Epoch  400  loss  0.4724670813626788 correct 50 avg time per epoch: 0.1501s
Epoch  410  loss  1.590503906232444 correct 48 avg time per epoch: 0.1313s
Epoch  420  loss  0.05801742356051693 correct 50 avg time per epoch: 0.0855s
Epoch  430  loss  0.016275568926112576 correct 50 avg time per epoch: 0.0853s
Epoch  440  loss  0.2590337127273972 correct 50 avg time per epoch: 0.0847s
Epoch  450  loss  0.2114533236149223 correct 49 avg time per epoch: 0.0854s
Epoch  460  loss  1.3423606426635846 correct 48 avg time per epoch: 0.0872s
Epoch  470  loss  0.048955714171329566 correct 48 avg time per epoch: 0.0854s
Epoch  480  loss  0.5685875363770083 correct 50 avg time per epoch: 0.0885s
Epoch  490  loss  0.04476384136031511 correct 49 avg time per epoch: 0.0863s



!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

 Epoch  0  loss  6.51040936369922 correct 28 avg time per epoch: 21.1487s
Epoch  10  loss  5.934883069677052 correct 49 avg time per epoch: 0.0877s
Epoch  20  loss  4.109442066664468 correct 43 avg time per epoch: 0.0874s
Epoch  30  loss  3.669008847380419 correct 49 avg time per epoch: 0.0880s
Epoch  40  loss  2.2158243124462564 correct 49 avg time per epoch: 0.0859s
Epoch  50  loss  1.8801367879385293 correct 49 avg time per epoch: 0.0859s
Epoch  60  loss  2.8546853715774008 correct 49 avg time per epoch: 0.0848s
Epoch  70  loss  0.7726617562563328 correct 50 avg time per epoch: 0.0930s
Epoch  80  loss  1.0319556800829646 correct 49 avg time per epoch: 0.1610s
Epoch  90  loss  0.4504550074205961 correct 50 avg time per epoch: 0.1183s
Epoch  100  loss  0.7735508806580451 correct 50 avg time per epoch: 0.0878s
Epoch  110  loss  0.586020420998616 correct 50 avg time per epoch: 0.0866s
Epoch  120  loss  0.5220622674291387 correct 50 avg time per epoch: 0.0872s
Epoch  130  loss  0.6741220177147089 correct 50 avg time per epoch: 0.0867s
Epoch  140  loss  0.49056951112846575 correct 50 avg time per epoch: 0.0874s
Epoch  150  loss  0.46579936635180974 correct 50 avg time per epoch: 0.0881s
Epoch  160  loss  0.4739710384817476 correct 50 avg time per epoch: 0.0858s
Epoch  170  loss  0.237254466282075 correct 50 avg time per epoch: 0.0865s
Epoch  180  loss  0.308810332648811 correct 50 avg time per epoch: 0.0862s
Epoch  190  loss  0.3114346576953332 correct 50 avg time per epoch: 0.0844s
Epoch  200  loss  0.0657240255620614 correct 50 avg time per epoch: 0.0940s
Epoch  210  loss  0.31558549974026895 correct 50 avg time per epoch: 0.1545s
Epoch  220  loss  0.30327264155879297 correct 50 avg time per epoch: 0.1203s
Epoch  230  loss  0.19445395816297087 correct 50 avg time per epoch: 0.0854s
Epoch  240  loss  0.32841787995970617 correct 50 avg time per epoch: 0.0861s
Epoch  250  loss  0.060041798349768014 correct 50 avg time per epoch: 0.0850s
Epoch  260  loss  0.3112026978938519 correct 50 avg time per epoch: 0.0855s
Epoch  270  loss  0.23701094471785827 correct 50 avg time per epoch: 0.0855s
Epoch  280  loss  0.08806794325944352 correct 50 avg time per epoch: 0.0881s
Epoch  290  loss  0.08229885670114126 correct 50 avg time per epoch: 0.0859s
Epoch  300  loss  0.03254276776602153 correct 50 avg time per epoch: 0.0861s
Epoch  310  loss  0.09285696140833086 correct 50 avg time per epoch: 0.0858s
Epoch  320  loss  0.07460861533804114 correct 50 avg time per epoch: 0.0846s
Epoch  330  loss  0.21434982757077964 correct 50 avg time per epoch: 0.0860s
Epoch  340  loss  0.2043797106975694 correct 50 avg time per epoch: 0.1530s
Epoch  350  loss  0.0685239944772239 correct 50 avg time per epoch: 0.1312s
Epoch  360  loss  0.3034879465817633 correct 50 avg time per epoch: 0.0868s
Epoch  370  loss  0.07619913946232452 correct 50 avg time per epoch: 0.0869s
Epoch  380  loss  0.02298647193525046 correct 50 avg time per epoch: 0.0850s
Epoch  390  loss  0.042937299641408695 correct 50 avg time per epoch: 0.0857s
Epoch  400  loss  0.13304622653083287 correct 50 avg time per epoch: 0.0874s
Epoch  410  loss  0.1978439371488575 correct 50 avg time per epoch: 0.0862s
Epoch  420  loss  0.042178840413064606 correct 50 avg time per epoch: 0.0858s
Epoch  430  loss  0.06205006417013895 correct 50 avg time per epoch: 0.0861s
Epoch  440  loss  0.12164542178013851 correct 50 avg time per epoch: 0.0845s
Epoch  450  loss  0.1230172677285078 correct 50 avg time per epoch: 0.0876s
Epoch  460  loss  0.19274728271441555 correct 50 avg time per epoch: 0.0858s
Epoch  470  loss  0.1551337104156188 correct 50 avg time per epoch: 0.1537s
Epoch  480  loss  0.0831090554282315 correct 50 avg time per epoch: 0.1334s
Epoch  490  loss  0.1738450623415352 correct 50 avg time per epoch: 0.0868s



!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

 Epoch  0  loss  6.40805120510812 correct 33 avg time per epoch: 20.7346s
Epoch  10  loss  4.138138554815679 correct 43 avg time per epoch: 0.1306s
Epoch  20  loss  5.185302925202857 correct 43 avg time per epoch: 0.0918s
Epoch  30  loss  3.03364884973628 correct 43 avg time per epoch: 0.0872s
Epoch  40  loss  2.856148029186451 correct 43 avg time per epoch: 0.0868s
Epoch  50  loss  1.9655744915601863 correct 44 avg time per epoch: 0.0868s
Epoch  60  loss  3.09771462206389 correct 45 avg time per epoch: 0.0852s
Epoch  70  loss  4.84501428079515 correct 44 avg time per epoch: 0.0864s
Epoch  80  loss  2.6348331693075844 correct 46 avg time per epoch: 0.0864s
Epoch  90  loss  1.033079164292102 correct 47 avg time per epoch: 0.0866s
Epoch  100  loss  2.7315774467588483 correct 48 avg time per epoch: 0.0860s
Epoch  110  loss  3.9945901510137163 correct 48 avg time per epoch: 0.0857s
Epoch  120  loss  1.2785327452864492 correct 48 avg time per epoch: 0.0856s
Epoch  130  loss  1.9543530575767052 correct 49 avg time per epoch: 0.1392s
Epoch  140  loss  1.1483231466507786 correct 48 avg time per epoch: 0.1424s
Epoch  150  loss  1.3744692178483457 correct 48 avg time per epoch: 0.0890s
Epoch  160  loss  1.1760650224459708 correct 48 avg time per epoch: 0.0866s
Epoch  170  loss  2.593901447878501 correct 49 avg time per epoch: 0.0903s
Epoch  180  loss  1.7293312086605157 correct 47 avg time per epoch: 0.0851s
Epoch  190  loss  0.3516517815903936 correct 49 avg time per epoch: 0.0856s
Epoch  200  loss  1.8722774341753818 correct 49 avg time per epoch: 0.0862s
Epoch  210  loss  0.42357214240601854 correct 50 avg time per epoch: 0.0862s
Epoch  220  loss  1.7416450386644584 correct 48 avg time per epoch: 0.0863s
Epoch  230  loss  1.0404434583541273 correct 49 avg time per epoch: 0.0877s
Epoch  240  loss  1.0248778146051842 correct 50 avg time per epoch: 0.0854s
Epoch  250  loss  1.2961978083127388 correct 50 avg time per epoch: 0.0849s
Epoch  260  loss  0.3167636201344281 correct 50 avg time per epoch: 0.1512s
Epoch  270  loss  1.0721360038015704 correct 50 avg time per epoch: 0.1379s
Epoch  280  loss  0.7631981038125419 correct 49 avg time per epoch: 0.0849s
Epoch  290  loss  0.051391544533150535 correct 50 avg time per epoch: 0.0851s
Epoch  300  loss  1.0528958050542903 correct 50 avg time per epoch: 0.0850s
Epoch  310  loss  0.40536772124407205 correct 50 avg time per epoch: 0.0853s
Epoch  320  loss  0.18100650126328308 correct 48 avg time per epoch: 0.0856s
Epoch  330  loss  0.2749784859097769 correct 50 avg time per epoch: 0.0852s
Epoch  340  loss  0.9672577191817266 correct 50 avg time per epoch: 0.0875s
Epoch  350  loss  0.21377833263153662 correct 50 avg time per epoch: 0.0857s
Epoch  360  loss  0.10937382486089468 correct 49 avg time per epoch: 0.0868s
Epoch  370  loss  1.2712920678628639 correct 50 avg time per epoch: 0.0848s
Epoch  380  loss  0.6952226379578664 correct 49 avg time per epoch: 0.0860s
Epoch  390  loss  1.291555109596627 correct 49 avg time per epoch: 0.1362s
Epoch  400  loss  0.15781093422585424 correct 50 avg time per epoch: 0.1412s
Epoch  410  loss  0.24579930005844772 correct 50 avg time per epoch: 0.0868s
Epoch  420  loss  0.8450877674703048 correct 49 avg time per epoch: 0.0852s
Epoch  430  loss  0.31092167621169076 correct 50 avg time per epoch: 0.0826s
Epoch  440  loss  1.010691487487201 correct 49 avg time per epoch: 0.0842s
Epoch  450  loss  0.9154829264277861 correct 49 avg time per epoch: 0.0844s
Epoch  460  loss  0.13918183997556713 correct 49 avg time per epoch: 0.0843s
Epoch  470  loss  0.7604042486434581 correct 50 avg time per epoch: 0.0857s
Epoch  480  loss  0.21378050153538147 correct 50 avg time per epoch: 0.0856s
Epoch  490  loss  0.13393248607681738 correct 50 avg time per epoch: 0.0836s


