"""Identify file aliases before writing a report beside its source data."""

from pathlib import Path
from typing import Iterable, Optional, Union

PathInput = Optional[Union[str, Path]]


def paths_overlap(paths: Iterable[PathInput]) -> bool:
    names = set()
    identities = set()
    for item in paths:
        if item is None:
            continue
        path = Path(item).expanduser().resolve()
        if path in names:
            return True
        names.add(path)
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        identity = info.st_dev, info.st_ino
        if identity in identities:
            return True
        identities.add(identity)
    return False
