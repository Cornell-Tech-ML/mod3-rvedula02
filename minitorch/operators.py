"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two floating point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the input value unchanged.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The same value as the input.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floating point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a floating point number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x < y, else 0.0.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x == y, else 0.0.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floating point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two floating point numbers are close to each other.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if |x - y| < 1e-2, else 0.0.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the rectified linear unit (ReLU) function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: e raised to the power of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the multiplicative inverse.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: 1 divided by x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm.

    Args:
    ----
        x (float): The input value.
        d (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of log(x) with respect to x.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the multiplicative inverse.

    Args:
    ----
        x (float): The input value.
        d (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of 1/x with respect to x.

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function.

    Args:
    ----
        x (float): The input value.
        d (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of ReLU(x) with respect to x.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply function fn to every element of iterable ls.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply to each element.
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        Iterable[float]: A new iterable with fn applied to each element of ls.

    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Apply function fn to every pair of elements from ls1 and ls2.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply to each pair of elements.
        ls1 (Iterable[float]): The first input iterable.
        ls2 (Iterable[float]): The second input iterable.

    Returns:
    -------
        Iterable[float]: A new iterable with fn applied to each pair of elements from ls1 and ls2.

    """
    result = []
    iter1 = iter(ls1)
    iter2 = iter(ls2)
    while True:
        try:
            x = next(iter1)
            y = next(iter2)
            result.append(fn(x, y))
        except StopIteration:
            break
    return result


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Apply function fn to every pair of elements from ls.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply to pairs of elements.
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        float: The result of applying fn to all elements in ls.

    """
    iterator = iter(ls)  # Create an iterator from the iterable
    try:
        result = next(iterator)
    except StopIteration:
        return 0.0
    for x in iterator:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate every element of iterable ls.

    Args:
    ----
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        Iterable[float]: A new iterable with each element negated.

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of ls1 and ls2.

    Args:
    ----
        ls1 (Iterable[float]): The first input iterable.
        ls2 (Iterable[float]): The second input iterable.

    Returns:
    -------
        Iterable[float]: A new iterable with corresponding elements from ls1 and ls2 added together.

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements of iterable ls.

    Args:
    ----
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        float: The sum of all elements in the iterable.

    """
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Multiply all elements of iterable ls.

    Args:
    ----
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        float: The product of all elements in the iterable.

    """
    return reduce(mul, ls)
