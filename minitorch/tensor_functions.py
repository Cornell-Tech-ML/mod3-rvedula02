"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple.

    Args:
    ----
        x (Any): The value to be wrapped.

    Returns:
    -------
        tuple: A tuple containing the value.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    """Base class for all functions in the autodifferentiation framework."""

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Compute the backward pass for the function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_out (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, ...]: The gradients with respect to the inputs.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Compute the forward pass for the function.

        Args:
        ----
            ctx (Context): The context for the operation.
            *inps (Tensor): The input tensors.

        Returns:
        -------
            Tensor: The result of the forward operation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history.

        Args:
        ----
            *vals (Tensor): The input tensors.

        Returns:
        -------
            Tensor: The output tensor after applying the function.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for negation.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for negation.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for inversion.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for inversion.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The sum of the two tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        return grad_output, grad_output


class All(Function):
    """Logical 'All' function."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements are true along the specified dimension.

        Args:
        ----
            ctx (Context): The context for the operation.
            a (Tensor): The input tensor.
            dim (Tensor): The dimension to check.

        Returns:
        -------
            Tensor: A tensor indicating if all elements are true.

        """
        ctx.save_for_backward(a, dim)
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Compute the backward pass for logical 'All'.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: A zero tensor with the same shape as the input.

        """
        a, dim = ctx.saved_values
        grad_a = a.f.zeros_like(a)
        return grad_a, None


class Mul(Function):
    """Multiplication function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The product of the two tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        t1, t2 = ctx.saved_values
        return grad_output * t2, grad_output * t1


class Sigmoid(Function):
    """Sigmoid activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The sigmoid of the input tensor.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (sigmoid_a,) = ctx.saved_values
        one = minitorch.Tensor.make(
            [1.0] * sigmoid_a._tensor.size, sigmoid_a.shape, backend=sigmoid_a.backend
        )
        one_minus_sigmoid = one - sigmoid_a
        sigmoid_derivative = sigmoid_a * one_minus_sigmoid
        grad_input = grad_output * sigmoid_derivative
        return grad_input


class ReLU(Function):
    """ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The ReLU of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Logarithm function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The exponential of the input tensor.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        (exp_t1,) = ctx.saved_values
        grad_input = grad_output.f.mul_zip(grad_output, exp_t1)
        return grad_input


class Sum(Function):
    """Sum function."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the forward pass for summation along specified dimensions.

        Args:
        ----
            ctx (Context): The context for the operation.
            a (Tensor): The input tensor.
            dim_tensor (Tensor): The dimensions to sum over.

        Returns:
        -------
            Tensor: The sum of the tensor along the specified dimensions.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for summation.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Less than comparison function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor indicating the result of the comparison.

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        return grad_output.zeros(grad_output.shape), grad_output.zeros(
            grad_output.shape
        )


class EQ(Function):
    """Equality comparison function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor indicating the result of the comparison.

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        return grad_output.zeros(grad_output.shape), grad_output.zeros(
            grad_output.shape
        )


class IsClose(Function):
    """Check if two tensors are close to each other."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for checking closeness.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor indicating if the two tensors are close.

        """
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        """No backward pass needed for closeness check.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[None, None]: No gradients are returned.

        """
        return None, None


class Permute(Function):
    """Permute the dimensions of a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Compute the forward pass for permuting dimensions.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.
            order (Tensor): The order to permute the dimensions.

        Returns:
        -------
            Tensor: The tensor with permuted dimensions.

        """
        ctx.save_for_backward(order)
        order2 = [int(order[i]) for i in range(order.size)]

        if len(order2) != len(t1.shape):
            raise ValueError(
                f"Permutation order length {len(order2)} does not match tensor dimensions {len(t1.shape)}."
            )

        if sorted(order2) != list(range(len(t1.shape))):
            raise ValueError(
                f"Invalid permutation order: {order2}. Must be a permutation of {list(range(len(t1.shape)))}."
            )

        permuted_shape = tuple(t1.shape[i] for i in order2)
        permuted_strides = tuple(t1._tensor.strides[i] for i in order2)
        return minitorch.Tensor.make(
            t1._tensor._storage, permuted_shape, permuted_strides, backend=t1.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for permuting dimensions.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        order_list = ctx.saved_values[0]
        inverse_order_storage = [0] * order_list.size
        for i in range(order_list.size):
            index = int(order_list[i])
            inverse_order_storage[index] = i

        inverse_order = tensor(inverse_order_storage)

        grad_input = Permute.apply(grad_output, inverse_order)
        zero_grad = zeros(order_list.shape, backend=order_list.backend)
        return (grad_input, zero_grad)


class View(Function):
    """View a tensor with a different shape."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Compute the forward pass for viewing a tensor.

        Args:
        ----
            ctx (Context): The context for the operation.
            a (Tensor): The input tensor.
            shape (Tensor): The new shape for the tensor.

        Returns:
        -------
            Tensor: The tensor viewed with the new shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"

        shape2 = [int(shape[i]) for i in range(shape.shape[0])]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the backward pass for viewing a tensor.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input and a placeholder.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    """Copy a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Identity function that makes a tensor contiguous.

        Args:
        ----
            ctx (Context): The context for the operation.
            a (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A contiguous copy of the input tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo the copy operation.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        return grad_output


class MatMul(Function):
    """Matrix multiplication function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for matrix multiplication.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for matrix multiplication.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the inputs.

        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend. Defaults to SimpleBackend.

    Returns:
    -------
        Tensor: A tensor filled with zeros.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend. Defaults to SimpleBackend.
        requires_grad (bool, optional): If True, the tensor will track gradients. Defaults to False.

    Returns:
    -------
        Tensor: A tensor filled with random values.

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data `ls` and shape `shape`.

    Args:
    ----
        ls (Any): Data for the tensor.
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend, optional): The tensor backend. Defaults to SimpleBackend.
        requires_grad (bool, optional): If True, the tensor will track gradients. Defaults to False.

    Returns:
    -------
        Tensor: The created tensor.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from `ls`.

    Args:
    ----
        ls (Any): Data for the tensor.
        backend (TensorBackend, optional): The tensor backend. Defaults to SimpleBackend.
        requires_grad (bool, optional): If True, the tensor will track gradients. Defaults to False.

    Returns:
    -------
        Tensor: The created tensor.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors
def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference gradient for a function.

    Args:
    ----
        f (Any): The function to differentiate.
        *vals (Tensor): The input tensors.
        arg (int, optional): The index of the argument to differentiate. Defaults to 0.
        epsilon (float, optional): The small change for finite difference. Defaults to 1e-6.
        ind (UserIndex): The index to apply the change.

    Returns:
    -------
        float: The estimated gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference.

    Args:
    ----
        f (Any): The function to check.
        *vals (Tensor): The input tensors to the function.

    Raises:
    ------
        AssertionError: If the gradients do not match.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
