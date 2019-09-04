from os import path
from datetime import datetime, timezone

import pandas as pd
# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras

from src.models.base.model import Model
from src.models.keras.build import _build_keras_model
from src.models.keras.load import \
    _load_keras_model, \
    KERAS_MODEL_CHECKPOINT_EXTENSION, \
    _get_latest_model_path, \
    _create_model_name

ModelCheckpoint = keras.callbacks.ModelCheckpoint


def _checkpoint_save(model_path: str) -> keras.callbacks.Callback:
    checkpoint_path_name = '{0}.{1}'.format(
        model_path,
        KERAS_MODEL_CHECKPOINT_EXTENSION
    )
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path_name,
        verbose=1
    )
    return cp_callback


class KerasModel(Model):
    _model_dir: str
    _model_path: str
    _keras_instance: keras.Model

    def __init__(self, model_dir: str, config: dict):
        super(KerasModel, self).__init__(model_dir, config)
        self._model_dir = model_dir
        model_name = datetime.now(timezone.utc).isoformat()
        self._model_path = _get_latest_model_path(self._model_dir, model_name)
        self._keras_instance = _load_keras_model(self._model_path)
        if self._keras_instance is None:
            self._keras_instance = _build_keras_model(config)
        self._checkpoint_cb = _checkpoint_save(self._model_path)

    def save(self):
        model_name = datetime.now(timezone.utc).isoformat()
        model_path = path.join(self._model_dir, _create_model_name(model_name))
        self._keras_instance.save(model_path)

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
