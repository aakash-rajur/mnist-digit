import pandas as pd
from src.features.mnist.pkg.constants.df_key import _construct_df_key
from src.features.mnist.pkg.constants.df_scale_key import _construct_df_scale_key


def transform_to_relative_coordinates(
        coordinates: pd.DataFrame,
        x_key: str, y_key: str,
        origin: (float, float)
) -> pd.DataFrame:
    """
    shifts origin of the provided coordinates, calculates their derivative of each
    ordinates, respective ordinate scale w.r.t to the coordinate which has longest
    euclidean distance from the origin, sort the points with the same distance in
    descending order
    :param coordinates: data frame of coordinates whose features needs to be
    calculated
    :param x_key: name of the column containing x ordinates
    :param y_key: name of the column containing x ordinates
    :param origin: coordinate that will be the new origin
    :return: data frame with calculated features
    """
    xc, yc = origin
    df = pd.concat([
        pd.Series(coordinates[x_key], name=x_key),
        pd.Series(coordinates[y_key], name=y_key),
    ], axis=1)

    dx = _construct_df_key(x_key)
    dy = _construct_df_scale_key(y_key)

    df[dx] = df[x_key] - xc
    df[dy] = df[y_key] - yc
    length = df['l'] = (df[dx] ** 2 + df[dy] ** 2) ** 0.5

    max_index = length.idxmax()
    x, y = df.loc[max_index, [dx, dy]]

    x_scaled = _construct_df_scale_key(x_key)
    y_scaled = _construct_df_scale_key(y_key)

    df[x_scaled] = df[dx] / x
    df[y_scaled] = df[dy] / y

    df = df.sort_values(by=['l'], ascending=False)

    return df
