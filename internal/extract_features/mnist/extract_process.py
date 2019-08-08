from environs import Env

from src.features.base.data_source.data_source import DataSource
from src.features.mnist.extract_process.extract_process import ExtractProcessMnist

MNIST_SCALE = 'MNIST_SCALE'


def _get_mnist_scale(env: Env) -> int:
    return env.int(MNIST_SCALE)


def get_mnist_instance(env: Env, precision: int) -> DataSource:
    return DataSource(
        processor=ExtractProcessMnist,
        processor_args={
            'scale': _get_mnist_scale(env),
            'cluster_size': precision
        },
        epoch_arg_name='source'
    )
