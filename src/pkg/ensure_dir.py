from os import path, makedirs


def ensure_dir(cwd: str, partial: str):
    output_dir = path.join(cwd, partial)
    makedirs(
        output_dir,
        exist_ok=True
    )
    return output_dir
