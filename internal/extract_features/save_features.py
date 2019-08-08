from os import path

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


def save_features(output_dir: str, features: dict, version: int = 1):
    for character, feature in features.items():
        file_name = _generate_feature_file_name(
            character,
            version
        )
        file_path = path.join(output_dir, file_name)
        feature.to_csv(
            file_path,
            sep=',',
            encoding='utf-8',
            index=False,
            header=False
        )
