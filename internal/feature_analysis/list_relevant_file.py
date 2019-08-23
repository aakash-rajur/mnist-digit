from os import path
from src.pkg.index_epochs import index_epochs
from src.pkg.config_constants import \
    FILE_META_EXTRACTOR_REGEX, \
    FILE_NAME_TEMPLATE_CHARACTER_PARTIAL


def list_relevant_files(cwd: str, mime: str, directory: str):
    epochs = index_epochs([mime], directory)
    meta = {}
    for file_path in epochs:
        file_name = path.basename(file_path)
        partials = FILE_META_EXTRACTOR_REGEX.search(file_name)
        character = partials.group(FILE_NAME_TEMPLATE_CHARACTER_PARTIAL)
        if character is None:
            continue
        files = meta[character] \
            if character in meta \
            else []
        files.append(path.join(cwd, file_path))
        meta[character] = files
    return meta
