from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from os import getcwd

from src.pkg.load_config import load_config
from src.pkg.load_env import load_env
from internal.model.build_models import construct_model_template, build_models


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)

    model_template = construct_model_template()
    model_map = build_models(model_template, project_directory, config)
    for character, model in model_map.items():
        print('CHARACTER {}'.format(character))
        model.describe()


if __name__ == '__main__':
    main()
