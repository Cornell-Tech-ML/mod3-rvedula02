import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of random points in a 2D space.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A list of tuples, each representing a point in 2D space with coordinates between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]

    """
    Represents a graph with points and their corresponding labels.

    Attributes:
        N: The number of points in the graph.
        X: A list of tuples, each representing a point in 2D space.
        y: A list of labels corresponding to each point in X.
    """


def simple(N: int) -> Graph:
    """Generates a simple dataset with binary labels based on the value of x_1.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset with binary labels based on a diagonal line.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset with binary labels based on two vertical lines.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset with binary labels based on XOR logic.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset with binary labels based on a circular boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a dataset with binary labels based on two intertwined spirals.

    Args:
    ----
        N: The total number of points to generate (should be even).

    Returns:
    -------
        A Graph object containing the generated points and their labels.

    """

    def x(t: float) -> float:
        """Calculates the x-coordinate of a point on the spiral.

        Args:
        ----
            t: The parameter for the spiral function.

        Returns:
        -------
            The x-coordinate of the point.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculates the y-coordinate of a point on the spiral.

        Args:
        ----
            t: The parameter for the spiral function.

        Returns:
        -------
            The y-coordinate of the point.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
