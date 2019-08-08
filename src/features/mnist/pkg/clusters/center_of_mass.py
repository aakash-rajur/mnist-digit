import numpy as np
from scipy import ndimage


def calculate_center_of_mass(data_points: np.ndarray) -> (float, float):
    return ndimage.center_of_mass(data_points)
