from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from os import getcwd, path

from src.models.keras.model import KerasModel
from src.pkg.load_config import load_config
from src.pkg.load_env import load_env
from src.pkg.ensure_dir import ensure_dir


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)

    character_0_config = config['0']
    input_dir = ensure_dir(project_directory, character_0_config['INPUT'])
    model_file_name = 'character_0.h5'
    model_path = path.join(input_dir, model_file_name)

    model = KerasModel(model_path=model_path, config=character_0_config)
    model.describe()


if __name__ == '__main__':
    main()
