import numpy as np


def normalize_drones(drones):
    lb = np.min([np.min((drone[0], drone[1])) for drone in drones])
    ub = np.max([np.max((drone[0], drone[1])) for drone in drones])
    normalized_drones = [
        (
            ((x1 - lb) / (ub - lb), (y1 - lb) / (ub - lb)),
            ((x2 - lb) / (ub - lb), (y2 - lb) / (ub - lb)),
            charge,
        )
        for ((x1, y1), (x2, y2), charge) in drones
    ]

    return normalized_drones, (lb, ub)


def denormalize_drones(normalized_drones, min_max_values):
    lb, ub = min_max_values

    # Denormalize coordinates
    denormalized_drones = [
        (
            ((x1 * (ub - lb) + lb), (y1 * (ub - lb) + lb)),
            ((x2 * (ub - lb) + lb), (y2 * (ub - lb) + lb)),
            charge,
        )
        for ((x1, y1), (x2, y2), charge) in normalized_drones
    ]

    return denormalized_drones


def normalize_blocks(blocks, min_max_values):
    lb, ub = min_max_values
    normalized_blocks = [
        [((x - lb) / (ub - lb), (y - lb) / (ub - lb)) for (x, y) in block]
        for block in blocks
    ]
    return normalized_blocks


def denormalize_blocks(blocks, min_max_values):
    lb, ub = min_max_values
    denormalized_blocks = [
        [((x) * (ub - lb) + lb, (y) * (ub - lb) + lb) for (x, y) in block]
        for block in blocks
    ]
    return denormalized_blocks
