from importlib import metadata


class _Distribution:
    def __init__(self, version: str):
        self.version = version


def get_distribution(name: str) -> _Distribution:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "0"
    return _Distribution(version)
