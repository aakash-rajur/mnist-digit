from os import getcwd, path

# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras


def _load_keras_model(model_path) -> keras.Model:
    if not path.exists(model_path):
        return None
    return keras.models.load_model(model_path)
