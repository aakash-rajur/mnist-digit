import pandas as pd

from src.pkg.load_features import load_features
from src.pkg.config_constants import FEATURE_FILE_EXTENSION


def load_training_data(data_dir: str) -> pd.DataFrame:
    return load_features(data_dir, FEATURE_FILE_EXTENSION)
