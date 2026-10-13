"""Model files are checked by size and SHA-256 against models.json."""

import hashlib
import json
from pathlib import Path

from petroknowledge import weights


def test_missing_resized_and_changed_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "same.gguf").write_bytes(b"abc")
    (tmp_path / "resized.gguf").write_bytes(b"abcd")
    (tmp_path / "changed.gguf").write_bytes(b"xyz")
    digest = hashlib.sha256(b"abc").hexdigest()
    entries = [
        {"file": "same.gguf", "bytes": 3, "sha256": digest},
        {"file": "gone.gguf", "bytes": 3, "sha256": digest},
        {"file": "resized.gguf", "bytes": 3, "sha256": digest},
        {"file": "changed.gguf", "bytes": 3, "sha256": digest},
    ]

    assert weights.problems(entries, tmp_path, quick=False) == [
        "missing: gone.gguf",
        "size differs: resized.gguf",
        "SHA-256 differs: changed.gguf",
    ]
    assert weights.problems(entries, tmp_path, quick=True) == [
        "missing: gone.gguf",
        "size differs: resized.gguf",
    ]


def test_the_manifest_is_read_as_plain_data(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"file": "m.gguf"}]))

    assert weights.read(path) == [{"file": "m.gguf"}]


def test_the_real_manifest_lists_each_file_once() -> None:
    entries = weights.read(Path(__file__).parent.parent / "models.json")
    files = [entry["file"] for entry in entries]

    assert len(files) == len(set(files))
    assert all(len(entry["sha256"]) == 64 for entry in entries)
