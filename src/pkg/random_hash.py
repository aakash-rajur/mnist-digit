from random import getrandbits


def generate_random_hash(length: int = None) -> str:
    bits = getrandbits(256)
    hexed = "%032x" % bits

    partial = hexed
    if length is None:
        return partial

    return partial[:length]
