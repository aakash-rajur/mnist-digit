from __future__ import absolute_import, division, print_function, unicode_literals, barry_as_FLUFL

import sys
from pathlib import Path

project_directory = str(Path('..').resolve())
sys.path.extend([project_directory])

# import tensorflow as tf
from tensorflow_core.python import keras
from tensorflow_core.python.keras import models, layers, optimizers, losses, metrics
from tensorflow_core.python.keras.engine.input_layer import Input


def main():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adagrad(),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.Accuracy()],
    )

    model.summary()


if __name__ == '__main__':
    main()
