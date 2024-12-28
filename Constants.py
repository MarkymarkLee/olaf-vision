import numpy as np


ACTIONS = [
    [1, 0.0, 0.0, 0.0],
    [-1, 0.0, 0.0, 0.0],
    [0.0, 1, 0.0, 0.0],
    [0.0, -1, 0.0, 0.0],
    [0.0, 0.0, 1, 0.0],
    [0.0, 0.0, -1, 0.0],
    [0.0, 0.0, 0.0, 1],
    [0.0, 0.0, 0.0, -1]
]

ACTION_PROMPTS = [
    "Move right.",
    "Move left.",
    "Move forward.",
    "Move backward.",
    "Move up.",
    "Move down.",
    "Close gripper.",
    "Open gripper."
]

NUMPY_ACTIONS = np.array(ACTIONS)
