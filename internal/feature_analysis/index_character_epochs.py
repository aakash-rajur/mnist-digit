import pandas as pd

PATH = 'PATH'
MIME = 'MIME'
INDEX_CHARACTER = 'character'
INDEX_EPOCH = 'epoch'


def _index_character_df(character: str, df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    df.insert(loc=0, column=INDEX_EPOCH, value=index)
    df.insert(loc=0, column=INDEX_CHARACTER, value=character)
    return df.set_index([INDEX_EPOCH, INDEX_CHARACTER])


def index_character_epochs(epochs: dict):
    compiled = None
    for character, df in epochs.items():
        indexed = _index_character_df(character, df)
        if compiled is None:
            compiled = indexed
        else:
            compiled = pd.concat([compiled, indexed])
    return compiled


def flatten_character_index(indexed: pd.DataFrame) -> pd.DataFrame:
    with_index = indexed.reset_index(level=INDEX_CHARACTER)
    columns_renamed = with_index.rename(
        columns={
            0: 'x',
            1: 'y',
            2: 'z',
        },
    )
    return columns_renamed


def get_distinct_characters(epochs: pd.DataFrame) -> list:
    distinct = list(
        epochs.index \
            .get_level_values(INDEX_CHARACTER) \
            .unique()
    )
    distinct.sort()
    return distinct
