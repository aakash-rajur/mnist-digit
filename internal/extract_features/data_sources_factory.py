from os import path, getcwd

from internal.extract_features.mnist.extract_process import get_mnist_instance
from src.pkg.load_env import load_env
import src.pkg.config_constants as constants

CWD = getcwd()

DATA_SOURCE_TYPE_TEMPLATE = {
    constants.TYPE_MNIST: {
        constants.ENV_DIR: path.join(CWD, 'internal', 'extract_features', 'mnist'),
        constants.DATA_SOURCE_INSTANCE: get_mnist_instance
    }
}


def data_sources_factory(precision: int) -> dict:
    data_source_factory = {}

    for data_source_type, data_source_template in DATA_SOURCE_TYPE_TEMPLATE.items():
        env_dir = data_source_template[constants.ENV_DIR]
        env = load_env(env_dir)

        get_instance = data_source_template[constants.DATA_SOURCE_INSTANCE]
        data_source_factory[data_source_type] = get_instance(env, precision)

    return data_source_factory
