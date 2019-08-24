import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from os import getcwd, path
import plotly.express as px
from environs import Env

from internal.feature_analysis.compile_epochs import compile_epochs
from internal.feature_analysis.index_character_epochs import index_character_epochs, flatten_character_index
from internal.feature_analysis.list_relevant_file import list_relevant_files
from internal.feature_analysis.reduce_dimensions import reduce_dimensions
from src.pkg.config_constants import PCA_COMPONENTS, INPUT
from src.pkg.load_config import load_config
from src.pkg.load_env import load_env
from src.pkg.run_process import run_process

PATH = 'PATH'
MIME = 'MIME'


def get_pca_dimensions(env: Env) -> int:
    return env.int(PCA_COMPONENTS)


def get_input(config: dict) -> (str, str):
    _input = config[INPUT]
    return _input[PATH], _input[MIME]


def main():
    cwd = getcwd()
    process_dir = cwd

    env = load_env(process_dir)
    config = load_config(process_dir, env)
    pca_dimensions = get_pca_dimensions(env)
    path_name, mime = get_input(config)

    meta = list_relevant_files(project_directory, mime, path.join(project_directory, path_name))
    epochs = compile_epochs(meta)

    indexed = index_character_epochs(epochs)
    reduced = reduce_dimensions(pca_dimensions, indexed)
    flattened = flatten_character_index(reduced)

    fig = px.scatter_3d(
        flattened,
        x='x',
        y='y',
        z='z',
        color='character',
        height=900,
        width=900,
    )
    fig.show()


if __name__ == '__main__':
    run_process(function=main)
