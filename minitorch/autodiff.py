from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
    ----
        f: Arbitrary function from n-scalar args to one value.
        *vals: n-float values $x_0 \ldots x_{n-1}$.
        arg: The number $i$ of the arg to compute the derivative.
        epsilon: A small constant.

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$.

    """
    vals_list = list(vals)
    curr = vals_list[arg]
    vals_list[arg] = curr + epsilon
    f_plus = f(*vals_list)

    vals_list[arg] = curr - epsilon
    f_minus = f(*vals_list)

    central_diff = (f_plus - f_minus) / (2 * epsilon)

    return central_diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable with respect to a given value.

        Args:
        ----
            x: The value with respect to which the derivative is to be accumulated.

        Returns:
        -------
            None.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable.

        Returns
        -------
            int: A unique identifier for the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph.

        Returns
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant in the computation graph.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of the variable's parents in the computation graph.

        Returns
        -------
            Iterable[Variable]: An iterable of the variable's parents.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the derivative of the output with respect to this variable.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing the variable and its derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable.

    Returns:
    -------
        Iterable of variables in topological order.

    """
    visited = set()
    order = []

    def dfs(v: Variable) -> None:
        if v.unique_id not in visited and not v.is_constant():
            visited.add(v.unique_id)
            for parent in v.parents:
                dfs(parent)
            order.append(v)

    dfs(variable)
    return reversed(order)  # Return in topological order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    grad_table = {variable.unique_id: deriv}
    # print(grad_table)

    sorted_variables = topological_sort(variable)
    for var in sorted_variables:
        d_output = grad_table[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_input in var.chain_rule(d_output):
                if parent.unique_id in grad_table:
                    grad_table[parent.unique_id] += d_input
                else:
                    grad_table[parent.unique_id] = d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        # print(f"Saving values for backward: {values}")
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for backward pass."""
        return self.saved_values
