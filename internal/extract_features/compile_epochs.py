from os import getcwd, path
from typing import List

from internal.extract_features.index_epochs import _index_epochs
import internal.extract_features.config_constants as constants


def compile_epochs(sources: List[dict]) -> dict:
    data_sources_compiled = {}

    for source in sources:
        epoch_input = path.join(getcwd(), source[constants.INPUT])
        mimes = source[constants.MIMES]
        epochs = _index_epochs(epoch_input, mimes)

        character = source[constants.CHARACTER]
        data_source_type = source[constants.DATA_SOURCE_TYPE]

        character_config = data_sources_compiled[character] \
            if character in data_sources_compiled \
            else {}
        compiled_epochs = character_config[data_source_type] \
            if data_source_type in character_config \
            else []
        compiled_epochs.extend(epochs)
        character_config[data_source_type] = compiled_epochs
        data_sources_compiled[character] = character_config

    return data_sources_compiled
