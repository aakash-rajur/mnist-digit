import pandas as pd
# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras

from src.models.base.model import Model
from src.models.keras.build import _build_keras_model
from src.models.keras.load import _load_keras_model


class KerasModel(Model):
    _model_path = str
    _keras_instance: keras.Model

    def __init__(self, model_path, config):
        self._keras_instance = _load_keras_model(model_path)
        if not self._keras_instance:
            self._keras_instance = _build_keras_model(config)

    def save(self):
        self._keras_instance.save(self._model_path)

    def train(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            batch_size: int,
            epoch: int,
            verbose: bool,
    ):
        pass

    def predict(self, *args, **kwargs):
        pass
