def _construct_df_key(key: str) -> str:
    """
    generate name of the df(x) column name
    :param key: name of the column whose derivative name is required
    :return: derivative name of f(x)
    """
    return 'd{0}'.format(key)
