import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from functools import partial
from os import getcwd

from environs import Env

from internal.extract_features.data_sources_factory import data_sources_factory
from internal.extract_features.extract_features import extract_features
from internal.extract_features.save_features import save_features
from src.pkg.config_constants import PRECISION, OUTPUT
from src.pkg.ensure_dir import ensure_dir
from src.pkg.load_config import load_config
from src.pkg.load_env import load_env
from src.pkg.random_hash import generate_random_hash
from src.pkg.run_process import run_process


def get_precision(env: Env):
    return env.int(PRECISION)


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)
    precision = get_precision(env)
    output_dir = ensure_dir(project_directory, config[OUTPUT])

    ds_factory = data_sources_factory(project_directory, precision)

    version = generate_random_hash(5)
    done_callback = partial(save_features, output_dir, version)

    extract_features(
        cwd=project_directory,
        config=config,
        data_source_factory=ds_factory,
        parallel=True,
        show_progress=True,
        done_callback=done_callback,
    )


if __name__ == '__main__':
    run_process(function=main)
