from pathlib import Path
import tomllib


def load_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    return config


def project_path(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path

