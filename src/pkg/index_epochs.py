import glob
from functools import reduce
from typing import List
from os import path


def __compile_mimes(compiled: List[str], file_type: str):
    compiled.extend(glob.glob(file_type))
    return compiled


def index_epochs(mimes: List[str], folder_path: str) -> List[str]:
    qualified_path_names = [path.join(folder_path, '*.{0}'.format(mime)) for mime in mimes]
    return reduce(__compile_mimes, qualified_path_names, [])
