from typing import Union
import numpy as np
from PIL import Image


def _from_image(img: Image, scale: int) -> np.ndarray:
    """
    convert image to numpy array
    :param img: image instance to convert from
    :param scale: scale the image appropriately in px
    :return: numpy array with values representing the strength/alpha
    of that pixel
    """
    data_points = np.array(img.resize((scale, scale), Image.ANTIALIAS))
    return np.where(data_points > 100, data_points, data_points > 100)


def _from_path(path: str, scale: int) -> np.ndarray:
    """
    convert image to numpy array
    :param path: path to read image instance to convert from
    :param scale: scale the image appropriately in px
    :return: numpy array with values representing the strength/alpha
    of that pixel
    """
    img = Image.open(path)
    return _from_image(img, scale)


def image_to_nd_array(img_or_path: Union[type(Image), str], scale: int) -> np.ndarray:
    """
    convert image to numpy array
    :param img_or_path: path of image instance or image instance itself to convert from
    :param scale: scale the image appropriately in px
    :return: numpy array with values representing the strength/alpha
    of that pixel
    """
    parse_image = _from_image if type(img_or_path) == Image else _from_path
    return parse_image(img_or_path, scale)
