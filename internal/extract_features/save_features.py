from os import path
import pandas as pd

from internal.extract_features.config_constants import \
    FILE_NAME_TEMPLATE_CHARACTER_PARTIAL, FILE_NAME_TEMPLATE_VERSION_PARTIAL, FILE_NAME_TEMPLATE


def _generate_feature_file_name(character: str, version: int):
    with_char = FILE_NAME_TEMPLATE.replace(
        FILE_NAME_TEMPLATE_CHARACTER_PARTIAL,
        character
    )
    with_version = with_char.replace(
        FILE_NAME_TEMPLATE_VERSION_PARTIAL,
        str(version)
    )
    return with_version


def save_features(output_dir: str, character: str, features: pd.DataFrame, version: int = 1):
    file_name = _generate_feature_file_name(
        character,
        version
    )
    file_path = path.join(output_dir, file_name)
    features.to_csv(
        file_path,
        sep=',',
        encoding='utf-8',
        index=False,
        header=False
    )
    print(
        'SAVED FEATURES FOR CHARACTER {0} IN {1}'.format(
            character,
            file_path
        )
    )
