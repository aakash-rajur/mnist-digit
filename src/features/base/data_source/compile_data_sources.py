from functools import reduce
from typing import List, Tuple, Any, Iterator
from concurrent.futures import ProcessPoolExecutor
from operator import methodcaller
import pandas as pd
from tqdm import tqdm

from src.features.base.data_source.data_source import DataSource
from src.features.base.extract_process.extract_process import ExtractProcess


def _ds_processor_reducer(compiled: List[ExtractProcess], data_source: (DataSource, List[str])):
    ds, epochs = data_source
    return compiled + list(ds.get_processors(epochs))


def _compile_data_source_serial(
        data_sources: List[Tuple[DataSource, List[Any]]],
        show_progress: bool
) -> Iterator[pd.Series]:
    """
    process every data_source serially
    :param data_sources: list of data_sources mapped with their respective variant
    of ExtractProcess that will process them
    :param show_progress: show ascii progress art?
    :return: features extracted after processing every epoch
    """
    processors = reduce(_ds_processor_reducer, data_sources, [])
    if show_progress:
        extracted_features = tqdm(
            map(methodcaller('extract_features'), processors),
            total=len(processors)
        )
    else:
        extracted_features = map(methodcaller('extract_features'), processors)
    return extracted_features


def _compile_data_source_parallel(
        data_sources: List[Tuple[DataSource, List[Any]]],
        show_progress: bool
) -> Iterator[pd.Series]:
    """
    process every data_source parallel
    :param data_sources: list of data_sources mapped with their respective variant
    of ExtractProcess that will process them
    :param show_progress: show ascii progress art?
    :return: features extracted after processing every epoch
    """
    processors = reduce(_ds_processor_reducer, data_sources, [])
    parallel = ProcessPoolExecutor()
    if show_progress:
        extracted_features = tqdm(
            parallel.map(methodcaller('extract_features'), processors),
            total=len(processors)
        )
    else:
        extracted_features = parallel.map(methodcaller('extract_features'), processors)
    return extracted_features


def compile_data_sources(
        data_sources: List[Tuple[DataSource, List[Any]]],
        parallel: bool,
        show_progress: bool
) -> pd.DataFrame:
    """
    will process several data_sources with their respective variants of ExtractProcess
    :param data_sources: list of data_sources mapped with their respective variant
    of ExtractProcess that will process them
    :param parallel: process data_sources parallel?
    :param show_progress: show ascii progress art?
    :return: features extracted after processing every epoch
    """
    executor = _compile_data_source_parallel if parallel else \
        _compile_data_source_serial
    compiled = pd.DataFrame(
        executor(
            data_sources=data_sources,
            show_progress=show_progress
        )
    )
    return compiled.dropna()
