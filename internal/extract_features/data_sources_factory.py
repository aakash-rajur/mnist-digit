from os import path

import src.pkg.config_constants as constants
from internal.extract_features.mnist.extract_process import get_mnist_instance
from src.pkg.load_env import load_env


def construct_data_source_type_template(base_dir: str) -> dict:
    return {
        constants.TYPE_MNIST: {
            constants.ENV_DIR: path.join(base_dir, 'internal', 'extract_features', 'mnist'),
            constants.DATA_SOURCE_INSTANCE: get_mnist_instance
        }
    }


def data_sources_factory(cwd: str, precision: int) -> dict:
    data_source_factory = {}
    data_source_type_template = construct_data_source_type_template(cwd)

    for data_source_type, data_source_template in data_source_type_template.items():
        env_dir = data_source_template[constants.ENV_DIR]
        env = load_env(env_dir)

        get_instance = data_source_template[constants.DATA_SOURCE_INSTANCE]
        data_source_factory[data_source_type] = get_instance(env, precision)

    return data_source_factory
