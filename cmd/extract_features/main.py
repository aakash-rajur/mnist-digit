from functools import partial
from os import getcwd, path

from environs import Env

from internal.extract_features.data_sources_factory import data_sources_factory
from internal.extract_features.extract_features import extract_features
from internal.extract_features.save_features import save_features
from src.pkg.config_constants import PRECISION
from src.pkg.get_output_dir import get_output_dir
from src.pkg.load_config import load_config
from src.pkg.load_env import load_env


def get_precision(env: Env):
    return env.int(PRECISION)


def main():
    cwd = getcwd()
    process_dir = path.join(cwd, 'cmd', 'extract_features')

    env = load_env(process_dir)
    config = load_config(process_dir, env)
    precision = get_precision(env)
    output_dir = get_output_dir(cwd, config)

    ds_factory = data_sources_factory(cwd, precision)
    extract_features(
        cwd=cwd,
        config=config,
        data_source_factory=ds_factory,
        parallel=True,
        show_progress=True,
        done_callback=partial(save_features, output_dir)
    )


if __name__ == '__main__':
    main()
