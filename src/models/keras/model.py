import pandas as pd
# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras

from src.models.base.model import Model
from src.models.keras.build import _build_keras_model
from src.models.keras.load import _load_keras_model


def _checkpoint_save(model_path: str) -> keras.callbacks.Callback:
    checkpoint_path_name = '{0}.ckpt'.format(model_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_name,
        verbose=1
    )
    return cp_callback


class KerasModel(Model):
    _model_path: str
    _keras_instance: keras.Model

    def __init__(self, model_path, config):
        self._model_path = model_path
        self._keras_instance = _load_keras_model(self._model_path)
        if not self._keras_instance:
            self._keras_instance = _build_keras_model(config)
        self._checkpoint_cb = _checkpoint_save(self._model_path)

    def save(self):
        self._keras_instance.save(self._model_path)

    def train(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            batch_size: int,
            epochs: int,
            verbose: bool,
    ):
        instance = self._keras_instance
        return instance.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[self._checkpoint_cb]
        )

    def predict(
            self,
            x: pd.DataFrame,
            batch_size: int,
            verbose: int,
    ):
        instance = self._keras_instance
        return instance.predict(
            x=x,
            batch_size=batch_size,
            verbose=verbose,
        )

    def describe(self):
        instance = self._keras_instance
        instance.summary()
