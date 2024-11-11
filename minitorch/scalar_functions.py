from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple.

    Args:
    ----
        x: A float or a tuple of floats.

    Returns:
    -------
        A tuple of floats.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables."""

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to the given scalar values.

        Args:
        ----
            vals: Scalar values to apply the function to.

        Returns:
        -------
            A Scalar object resulting from the function application.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Computes the backward pass for the function."""
        if not hasattr(cls, "backward"):
            raise NotImplementedError(
                f"{cls.__name__} must implement a backward method."
            )
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Computes the forward pass for the function.

        Args:
        ----
            ctx: The context for this operation.
            inps: Input values for the function.

        Returns:
        -------
            The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx: The context for this operation.
            a: First operand.
            b: Second operand.

        Returns:
        -------
            The result of the addition.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivatives with respect to the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The logarithm of the input value.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for logarithm.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the logarithm with respect to the input.

        """
        (a,) = ctx.saved_values
        return (operators.log_back(a, d_output),)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx: The context for this operation.
            a: First operand.
            b: Second operand.

        Returns:
        -------
            The product of the two operands.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivatives with respect to the inputs.

        """
        (a, b) = ctx.saved_values
        return (d_output * b, d_output * a)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The inverse of the input value.

        """
        ctx.save_for_backward(a)
        return 1 / float(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for inverse.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the inverse with respect to the input.

        """
        (a,) = ctx.saved_values
        return (-d_output / (a**2),)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The negation of the input value.

        """
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for negation.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the negation with respect to the input.

        """
        return (-d_output,)  # Return as a tuple


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The sigmoid of the input value.

        """
        result = 1 / (1 + operators.exp(-a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the sigmoid with respect to the input.

        """
        (sigmoid_result,) = ctx.saved_values
        return (d_output * sigmoid_result * (1 - sigmoid_result),)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The ReLU of the input value.

        """
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for ReLU.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the ReLU with respect to the input.

        """
        if not ctx.saved_values:
            return (0,)
        return (d_output,) if ctx.saved_values[0] > 0 else (0,)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential.

        Args:
        ----
            ctx: The context for this operation.
            a: The input value.

        Returns:
        -------
            The exponential of the input value.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for exponential.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the exponential with respect to the input.

        """
        (a,) = ctx.saved_values
        return (d_output * operators.exp(a),)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than.

        Args:
        ----
            ctx: The context for this operation.
            a: First operand.
            b: Second operand.

        Returns:
        -------
            The result of the less than comparison.

        """
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivatives with respect to the inputs.

        """
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality.

        Args:
        ----
            ctx: The context for this operation.
            a: First operand.
            b: Second operand.

        Returns:
        -------
            The result of the equality comparison.

        """
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality.

        Args:
        ----
            ctx: The context for this operation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivatives with respect to the inputs.

        """
        return (0.0, 0.0)
