from json import load
from os import path

from environs import Env

from src.pkg.config_constants import CONFIG_PATH


def load_config(process_dir: str, env: Env):
    config_path = path.join(process_dir, env.str(CONFIG_PATH))
    file = open(config_path)
    return load(file)
