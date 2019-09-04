from os import path

import pandas as pd

from src.pkg.config_constants import FEATURE_FILE_EXTENSION


def save_features(output_dir: str, version: str, features: pd.DataFrame):
    file_name = '{0}.{1}'.format(
        version,
        FEATURE_FILE_EXTENSION
    )
    file_path = path.join(output_dir, file_name)
    features.to_csv(
        path_or_buf=file_path,
        sep=',',
        encoding='utf-8',
        index=False,
        header=False,
    )
    print(
        'SAVED FEATURES IN {0}'.format(file_path)
    )
