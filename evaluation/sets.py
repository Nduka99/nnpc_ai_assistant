"""The registry of keyed question sets and their saved runs.

Each set's files live in git-ignored `data/eval/<set>/`: answer keys must never
reach a public repository, or a future model could be trained on them. The
committed registry (`evaluation/sets.json`) records every file's SHA-256, so a
changed key or run is caught before anything is scored. A registered hash never
changes: files can be added to a set (a new run, a new grading sitting), but a
different key is a different set with a new name.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("data/eval")
REGISTRY = Path("evaluation/sets.json")


def file_hashes(folder: Path) -> dict[str, str]:
    """Relative POSIX path -> SHA-256 of every file under the folder, sorted."""
    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def read_registry(registry: Path) -> dict[str, Any]:
    if not registry.exists():
        return {"sets": {}}
    data: dict[str, Any] = json.loads(registry.read_text(encoding="utf-8"))
    return data


def problems(name: str, root: Path, registry: Path) -> list[str]:
    """Every difference between a set's files and its registered hashes."""
    registered = read_registry(registry)["sets"].get(name)
    if registered is None:
        return [f"{name}: not registered"]
    found = file_hashes(root / name)
    listed = registered["files"]
    missing = []
    changed = []
    for path, digest in listed.items():
        if path not in found:
            missing.append(f"{name}/{path}: missing")
        elif found[path] != digest:
            changed.append(f"{name}/{path}: changed since it was registered")
    unregistered = []
    for path in found:
        if path not in listed:
            unregistered.append(f"{name}/{path}: not registered")
    return missing + changed + unregistered


def register(name: str, description: str, root: Path, registry: Path) -> list[str]:
    """Record the hashes of the set's new files; return the paths added.

    Refuses when a registered file is missing or changed, so a registered key or
    run can never be replaced in place.
    """
    changed = [p for p in problems(name, root, registry) if "not registered" not in p]
    if changed:
        raise ValueError("; ".join(changed))
    data = read_registry(registry)
    entry = data["sets"].setdefault(name, {"description": description, "files": {}})
    added = [path for path in file_hashes(root / name) if path not in entry["files"]]
    entry["files"] = file_hashes(root / name)
    data["sets"] = dict(sorted(data["sets"].items()))
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    registry.write_text(text, encoding="utf-8", newline="\n")
    return added


def require(name: str, root: Path, registry: Path) -> Path:
    """The set's folder, only if every file matches its registered hash."""
    found = problems(name, root, registry)
    if found:
        raise ValueError("; ".join(found))
    return root / name
