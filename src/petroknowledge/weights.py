"""The models this project uses, listed in models.json at the repository root: each
file's source, revision, size, SHA-256, licence and use. The files themselves live
in the git-ignored models/ folder (large, and published elsewhere); `petroknowledge
models` checks that every listed file is there and unchanged.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = Path("models.json")
FOLDER = Path("models")


def sha256(path: Path) -> str:
    """The file's SHA-256, read in chunks (model files run to tens of gigabytes)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def problems(entries: list[dict[str, Any]], folder: Path, quick: bool) -> list[str]:
    """What is missing or changed: every file's size, and its SHA-256 unless `quick`."""
    found = []
    for entry in entries:
        path = folder / entry["file"]
        if not path.is_file():
            found.append(f"missing: {entry['file']}")
        elif path.stat().st_size != entry["bytes"]:
            found.append(f"size differs: {entry['file']}")
        elif not quick and sha256(path) != entry["sha256"]:
            found.append(f"SHA-256 differs: {entry['file']}")
    return found


def read(manifest: Path = MANIFEST) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads(manifest.read_text(encoding="utf-8"))
    return entries
