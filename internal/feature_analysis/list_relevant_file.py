from os import path

import pandas as pd

from src.pkg.index_epochs import index_epochs


def load_features(base_dir: str, dir: str, mime: str):
    files_dir = path.join(base_dir, dir)
    feature_files = index_epochs([mime], files_dir)

    features = pd.concat(
        map(
            lambda file_path: pd.read_csv(
                filepath_or_buffer=file_path,
                header=None,
            ),
            feature_files
        )
    )

    return features
