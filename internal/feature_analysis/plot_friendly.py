import pandas as pd
from internal.feature_analysis.index_character_epochs import INDEX_CHARACTER


def construct_plot_friendly(characters: list, colors: list, coordinates: pd.DataFrame) -> list:
    friendly = []
    for index, character in enumerate(characters):
        each = (
            character,
            colors[index],
            coordinates.xs(
                key=character,
                level=INDEX_CHARACTER
            )
        )
        friendly.append(each)
    return friendly
