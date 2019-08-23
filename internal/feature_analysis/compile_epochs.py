import pandas as pd


def compile_epochs(meta: dict):
    compiled = {}
    for character, file_paths in meta.items():
        partials = map(
            lambda each: pd.read_csv(each, header=None),
            file_paths
        )
        compiled[character] = pd.concat(
            partials,
            ignore_index=True
        )
    return compiled
