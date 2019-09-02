from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from os import getcwd, path

# noinspection PyUnresolvedReferences,PyPackageRequirements
from tensorflow.python import keras

from src.pkg.load_config import load_config
from src.pkg.load_env import load_env
from src.pkg.ensure_dir import ensure_dir

MODEL = "MODEL"
INPUT = "INPUT"
OUTPUT = "OUTPUT"
OUTPUT_ACTIVATION = "OUTPUT_ACTIVATION"
LAYERS = "LAYERS"
LAYER_UNITS = "UNITS"
LAYER_ACTIVATION = "ACTIVATION"
LOSS = "LOSS"
OPTIMIZER = "OPTIMIZER"
METRICS = "METRICS"


def build_keras_model(model_config: dict) -> keras.Model:
    input_size = model_config[INPUT]
    input_layer = keras.Input(shape=(input_size,))

    layers = input_layer
    for layer in model_config[LAYERS]:
        layer_units = layer[LAYER_UNITS]
        layer_activation = layer[LAYER_ACTIVATION]

        layers = keras.layers.Dense(
            units=layer_units,
            activation=layer_activation
        )(layers)

    output_size = model_config[OUTPUT]
    output_activation = model_config[OUTPUT_ACTIVATION]
    output_layer = keras.layers.Dense(
        units=output_size,
        activation=output_activation
    )(layers)

    model = keras.Model(
        inputs=input_layer,
        outputs=output_layer
    )

    optimizer = model_config[OPTIMIZER]
    loss = model_config[LOSS]
    metrics = model_config[METRICS]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def load_keras_model(model_path) -> keras.Model:
    if not path.exists(model_path):
        return None
    return keras.models.load_model(model_path)


def save_keras_model(model: keras.Model, model_path):
    model.save(model_path)


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)

    character_0_config = config['0']
    input_dir = ensure_dir(project_directory, character_0_config['INPUT'])
    model_file_name = 'character_0.h5'

    model_path = path.join(project_directory, input_dir, model_file_name)

    model_0 = load_keras_model(model_path)

    if model_0 is None:
        model_0 = build_keras_model(character_0_config[MODEL])

    model_0.summary()

    model_0.save(model_path)


if __name__ == '__main__':
    main()
