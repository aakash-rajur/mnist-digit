from os import path, makedirs

from src.pkg.config_constants import OUTPUT


def get_output_dir(cwd: str, config: dict):
    output_dir = path.join(cwd, config[OUTPUT])
    makedirs(
        output_dir,
        exist_ok=True
    )
    return output_dir
