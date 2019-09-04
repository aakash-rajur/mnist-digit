import sys
from pathlib import Path

from internal.extract_features.save_features import save_features
from src.features.base.data_source.compile_data_sources import compile_data_sources
from src.features.base.data_source.data_source import DataSource
from src.features.mnist.extract_process.extract_process import ExtractProcessMnist
from src.pkg.config_constants import CHARACTER
from src.pkg.random_hash import generate_random_hash
from src.pkg.run_process import run_process

project_directory = str(Path('..').resolve())
sys.path.extend([project_directory])

from os import path
from ntpath import basename
import re
from functools import reduce

from src.pkg.index_files import index_files

infer_character = re.compile('(\d)_.*')

MNIST_DATA_SOURCE = DataSource(
    processor=ExtractProcessMnist,
    processor_args={
        'scale': 16,
        'cluster_size': 8
    },
    epoch_arg_name='source'
)


def _infer_meta(reduced: dict, file_path: str) -> dict:
    file_name = basename(file_path)
    match = infer_character.match(file_name)
    character = match[1]
    files = reduced[character] \
        if character in reduced \
        else []
    files.append(file_path)
    reduced[character] = files
    return reduced


def main():
    validation_image_dir = path.join(project_directory, 'data/external/mnist/validation')
    image_files = index_files(['jpg', 'jpeg'], validation_image_dir)
    sorted_files = reduce(_infer_meta, image_files, {})

    features = None
    for character, files in sorted_files.items():
        partial = compile_data_sources(
            [(MNIST_DATA_SOURCE, files)],
            True,
            True,
        )
        partial[CHARACTER] = character
        if features is None:
            features = partial
        else:
            features = features.append(partial, ignore_index=True)

    version = generate_random_hash(5)
    output_dir = path.join(project_directory, 'data/processed/validation')
    save_features(output_dir, version, features)


if __name__ == '__main__':
    run_process(main)
