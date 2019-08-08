from typing import Union
from PIL import Image
import pandas as pd
import numpy as np

from src.features.base.extract_process.extract_process import ExtractProcess
from src.features.mnist.pkg.image.image_to_nd_array import image_to_nd_array
from src.features.mnist.pkg.image.nd_array_to_df import img_nd_array_to_df
from src.features.mnist.pkg.clusters.center_of_mass import calculate_center_of_mass
from src.features.mnist.pkg.clusters.k_mean_clusters import calculate_k_means_clusters
from src.features.mnist.pkg.clusters.relative_coordinates import transform_to_relative_coordinates
from src.features.mnist.pkg.constants.df_scale_key import _construct_df_scale_key


def _extract_features(img_or_path: Union[type(Image), str], scale: int, cluster_size: int) -> pd.Series:
    """
    Extract features from an image:
    1. identify points that most probably represent the curves themselves.
    2. loose absolute referencing of these points by finding the center
       of mass all points
    3. shift the origin of the points identified in step 1.
    4. find the scale of each ordinate w.r.t to the point that is placed
       farthest in euclidean terms
    :param img_or_path: image to extract features from
    :param scale: scale dimension if pxs before extracting any features.
    :param cluster_size: no of coordinates to calculate in step 1.
    :return: pandas series representing the features of the character in image
    """
    x_key, y_key = 'x', 'y'
    image = image_to_nd_array(img_or_path, scale)

    center_of_mass = calculate_center_of_mass(image)
    coordinates = img_nd_array_to_df(
        image,
        x_key, y_key,
        lambda each: each > 100
    )
    clusters = calculate_k_means_clusters(coordinates, cluster_size, x_key, y_key)

    relative_coordinates = transform_to_relative_coordinates(
        clusters,
        x_key, y_key,
        center_of_mass
    )
    a = relative_coordinates[_construct_df_scale_key(x_key)]
    b = relative_coordinates[_construct_df_scale_key(y_key)]

    return pd.Series(np.column_stack((a, b)).flatten())


class ExtractProcessMnist(ExtractProcess):
    source: Union[type(Image), str]
    scale: int
    cluster_size: int

    def __init__(self, source: Union[type(Image), str], scale: int, cluster_size: int) -> None:
        super().__init__()
        self.source = source
        self.scale = scale
        self.cluster_size = cluster_size

    def extract_features(self) -> pd.Series:
        return _extract_features(
            self.source,
            self.scale,
            self.cluster_size
        )
