from os import path

# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras

from src.pkg.index_epochs import index_epochs

KERAS_MODEL_EXTENSION = "h5"
KERAS_MODEL_CHECKPOINT_EXTENSION = "ckpt"


def _create_model_name(model_name: str) -> str:
    return '{0}.{1}'.format(model_name, KERAS_MODEL_EXTENSION)


def _get_latest_model_path(model_dir: str, file_name: str) -> str:
    all_models = index_epochs([KERAS_MODEL_EXTENSION], model_dir)
    no_of_models = len(all_models)
    if no_of_models == 0:
        return path.join(model_dir, _create_model_name(file_name))

    sorted_models = sorted(all_models, reverse=True)
    latest_model = sorted_models[0]

    return latest_model


def _load_keras_model(model_path) -> keras.Model:
    if model_path is None or not path.exists(model_path):
        return None
    return keras.models.load_model(model_path)
