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


def _build_keras_model(model_config: dict) -> keras.Model:
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
