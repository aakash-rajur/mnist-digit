from os import getcwd, path, makedirs
from environs import Env
from json import load

from src.pkg.load_env import load_env
from internal.extract_features.config_constants import CONFIG_PATH, PRECISION, OUTPUT
from internal.extract_features.data_sources_factory import data_sources_factory
from internal.extract_features.extract_features import extract_features
from internal.extract_features.save_features import save_features


def load_config(process_dir: str, env: Env):
    config_path = path.join(process_dir, env.str(CONFIG_PATH))
    file = open(config_path)
    return load(file)


def get_precision(env: Env):
    return env.int(PRECISION)


def get_output_dir(cwd: str, config: dict):
    output_dir = path.join(cwd, config[OUTPUT])
    makedirs(
        output_dir,
        exist_ok=True
    )
    return output_dir


def main():
    cwd = getcwd()
    process_dir = path.join(cwd, 'cmd', 'extract_features')

    env = load_env(process_dir)
    config = load_config(process_dir, env)
    precision = get_precision(env)
    output_dir = get_output_dir(cwd, config)

    ds_factory = data_sources_factory(precision)
    features = extract_features(
        config=config,
        data_source_factory=ds_factory,
        parallel=True,
        show_progress=True
    )

    save_features(
        output_dir=output_dir,
        features=features
    )


if __name__ == '__main__':
    main()
