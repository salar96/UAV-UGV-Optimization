import numpy as np
from typing import List, Tuple


my_inf = 1e6  # a large value used for penalty function


def my_log(m):
    return np.log(m, out=np.zeros_like(m), where=(m > 0))


def penalty(x):
    x = np.clip(x, None, 2.0)
    a = 10
    z0 = 200
    x_off = 0.05
    y = a * np.log(
        (np.log(np.exp(1 / z0) + np.exp(z0 * x - z0 * x_off)))
        / (np.log(np.exp(1 / z0) + np.exp(-z0)))
    )
    return y


# def penalty(x):
#     mask = x<=0
#     y = my_inf*(x**2)
#     y[mask] = 0
#     return y


def create_block(
    type: str,
    center: Tuple[float, float],
    length: float,
    distortion: str = "none",
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """
    Creates a 2D shape with specified type, center, and length.
    Adds distortion like skew or rotation if specified.

    Parameters:
        type (str): Shape type ("hexagon", "square", "triangle", etc.).
        center (Tuple[float, float]): Coordinates of the center of the shape.
        length (float): Length of edges or characteristic size.
        distortion (str): Type of distortion ("none", "skewed", "rotated").
        seed (int): Random seed for reproducibility.

    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates of the shape vertices.
    """
    np.random.seed(seed)
    cx, cy = center
    vertices = []

    if type == "hexagon":
        # Generate vertices of a regular hexagon
        for i in range(6):
            angle = 2 * np.pi * i / 6
            x = cx + length * np.cos(angle)
            y = cy + length * np.sin(angle)
            vertices.append((x, y))

    elif type == "square":
        # Generate vertices of a square
        half = length / 2
        vertices = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]

    elif type == "triangle":
        # Generate vertices of an equilateral triangle
        for i in range(3):
            angle = 2 * np.pi * i / 3 - np.pi / 6  # Start at the top
            x = cx + length * np.cos(angle)
            y = cy + length * np.sin(angle)
            vertices.append((x, y))

    else:
        raise ValueError(f"Unsupported shape type: {type}")

    # Apply distortion
    if distortion == "skewed":
        skew_factor = np.random.uniform(-1.5, 1.5)
        vertices = [(x + skew_factor * y, y) for x, y in vertices]

    elif distortion == "rotated":
        angle = np.random.uniform(0, 2 * np.pi)
        vertices = [
            (
                cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle),
                cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle),
            )
            for x, y in vertices
        ]

    elif distortion != "none":
        raise ValueError(f"Unsupported distortion type: {distortion}")

    return vertices
