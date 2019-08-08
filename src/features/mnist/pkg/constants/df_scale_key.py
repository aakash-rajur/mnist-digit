def _construct_df_scale_key(key: str) -> str:
    """
    generate scale name of the df(x) column name
    :param key: name of the column whose derivative name is required
    :return: scaled derivative name of f(x)
    """
    return '{0}_scale'.format(key)
