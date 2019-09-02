import sys
from pathlib import Path

project_directory = str(Path('..').resolve())
sys.path.extend([project_directory])

from os import getcwd

from src.models.keras.model import KerasModel
from src.pkg.load_config import load_config
from src.pkg.load_env import load_env


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)

    print(config)


if __name__ == '__main__':
    main()
