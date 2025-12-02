from os import PathLike
from pathlib import Path

from pandas import DataFrame, read_csv


def load_csv(source: str | PathLike | DataFrame) -> DataFrame:
    # if it already is a Dataframe, return it
    if isinstance(source, DataFrame):
        return source.copy()

    # Otherwise, assume it is a file path
    if not isinstance(source, (str, bytes, PathLike)):
        raise TypeError(f"source must be a file path or a pandas dataframe: {type(source)}")

    # Check if file exists
    if not Path(source).exists():
        raise FileNotFoundError(f"File {source} not found.")

    # Load CSV into DataFrame and return items-endpoint DataFrame
    return read_csv(source)
