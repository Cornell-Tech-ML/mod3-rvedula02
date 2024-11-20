from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, parameters: Sequence[Parameter]):
        """Initialize the optimizer with parameters.

        Args:
        ----
            parameters (Sequence[Parameter]): The parameters to optimize.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initialize the SGD optimizer.

        Args:
        ----
            parameters (Sequence[Parameter]): The parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset the gradients of all parameters to zero."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Update parameters based on gradients."""
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
