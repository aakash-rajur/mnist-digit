import sys
from pathlib import Path

project_directory = str(Path('../..').resolve())
sys.path.extend([project_directory])

from os import getcwd
import plotly.express as px
from environs import Env

from src.pkg.load_features import load_features
from internal.feature_analysis.reduce_dimensions import reduce_dimensions
from src.pkg.config_constants import PCA_COMPONENTS, INPUT, CHARACTER
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
    input_dir_name, mime = get_input(config)

    features = load_features(project_directory, input_dir_name, mime)
    data_points = features.iloc[:, :-1]

    reduced = reduce_dimensions(pca_dimensions, data_points)
    reduced[CHARACTER] = features.iloc[:, -1].astype(str)
    reduced = reduced.rename(columns={0: 'x', 1: 'y', 2: 'z'})

    fig = px.scatter_3d(
        reduced,
        x='x',
        y='y',
        z='z',
        color=CHARACTER,
        height=900,
        width=900,
    )
    fig.show()


if __name__ == '__main__':
    run_process(function=main)
