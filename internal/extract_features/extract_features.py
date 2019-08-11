from typing import Callable, Any
import pandas as pd

from internal.extract_features.compile_epochs import compile_epochs
from src.features.base.data_source.compile_data_sources import compile_data_sources
import internal.extract_features.config_constants as constants


def _character_config(compiled_epochs: dict, data_source_factory: dict) -> dict:
    friendly = {}
    for character, config in compiled_epochs.items():
        character_data_sources = []
        for data_source_type, epochs in config.items():
            data_source = data_source_factory[data_source_type]
            character_data_sources.append((
                data_source,
                epochs
            ))
        friendly[character] = character_data_sources

    return friendly


def extract_features(
        config: dict,
        data_source_factory: dict,
        parallel: bool,
        show_progress: bool,
        done_callback: Callable[[str, pd.DataFrame], Any]
):
    sources = config[constants.SOURCES]
    compiled_epochs = compile_epochs(sources)

    character_configs = _character_config(compiled_epochs, data_source_factory)

    for character, data_sources in character_configs.items():
        print('PROCESSING DATA SOURCES FOR CHARACTER: {0}'.format(character))
        done_callback(character, compile_data_sources(data_sources, parallel, show_progress))

