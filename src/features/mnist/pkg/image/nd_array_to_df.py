from typing import Callable
import pandas as pd
import numpy as np


def img_nd_array_to_df(
        matrix: np.ndarray,
        x_key: str, y_key: str,
        criteria: Callable[[any], bool]
) -> pd.DataFrame:
    """
    converts an numpy matrix to an pandas data frame.
    :param matrix: numpy matrix representing the image
    :param x_key: column name representing the x ordinate
    :param y_key: column name representing the y ordinate
    :param criteria: filter out coordinates upon a predicate
    :return: a pandas data frame having filtered coordinates
    """
    (height, width) = matrix.shape
    compiled = []

    for row in range(height):
        for col in range(width):
            value = matrix[row][col]
            if criteria(value):
                compiled.append({
                    x_key: row,
                    y_key: col
                })

    return pd.DataFrame(compiled)
