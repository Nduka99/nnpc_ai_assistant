"""Run every check a change must pass before it is committed.

    uv run python scripts/check.py

The same command runs on a developer's machine and on the CI runner, so both check
exactly the same things. Every step runs, and all failures are listed at the end.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

# Files describing how the work was done may name the AI tools used (Rule 3);
# no other shipped file may. "Cursor" is left out: database drivers have cursors.
METHOD_FILES = {"README.md"}
# The scanner and its test hold the patterns they search for, so they are skipped.
SCANNER_FILES = {"scripts/check.py", "tests/test_check_script.py"}
AGENT_NAME = re.compile(r"\b(claude|codex|gemini|copilot)\b", re.IGNORECASE)
ATTRIBUTION = re.compile(
    r"co-authored-by|generated with \[?(claude|codex|gemini|copilot)", re.IGNORECASE
)

PYTHON = sys.executable
SOURCES = ["src", "evaluation", "tests", "scripts"]
# The web app's checks run through npm (npm.cmd on Windows); `npm ci` in web/ installs
# what they need from package-lock.json. The end-to-end run builds the app first.
NPM = shutil.which("npm") or "npm"
STEPS = [
    ("format", [PYTHON, "-m", "ruff", "format", "--check", *SOURCES]),
    ("lint", [PYTHON, "-m", "ruff", "check", "."]),
    ("types", [PYTHON, "-m", "mypy", *SOURCES]),
    ("tests", [PYTHON, "-m", "pytest", "-q"]),
    ("web lint", [NPM, "--prefix", "web", "run", "lint"]),
    # Impeccable's detector: deterministic checks for AI-default design (Rule 6).
    ("web design", [NPM, "--prefix", "web", "run", "design"]),
    ("web types", [NPM, "--prefix", "web", "run", "typecheck"]),
    ("web tests", [NPM, "--prefix", "web", "test"]),
    ("web e2e", [NPM, "--prefix", "web", "run", "e2e"]),
]


def attribution_problems(texts: dict[str, str]) -> list[str]:
    """Paths (or commit messages) that break Rule 3, with the reason."""
    problems = []
    for path, text in texts.items():
        if path in SCANNER_FILES:
            continue
        if ATTRIBUTION.search(text):
            problems.append(f"{path}: AI attribution")
        elif AGENT_NAME.search(text) and path not in METHOD_FILES:
            problems.append(f"{path}: names an AI tool outside a method section")
    return problems


def git_lines(*args: str) -> list[str]:
    output = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return [line for line in output.stdout.splitlines() if line]


def files_to_commit() -> dict[str, str]:
    """Tracked files plus new files git would add; ignored files never ship, so they
    are not scanned."""
    texts = {}
    for path in git_lines("ls-files", "--cached", "--others", "--exclude-standard"):
        if Path(path).is_file():
            texts[path] = Path(path).read_bytes().decode("utf-8", errors="ignore")
    return texts


def unpushed_messages() -> dict[str, str]:
    """Commit messages not yet on the remote's main branch (every commit while there
    is no remote, none before the first commit)."""
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD"], capture_output=True
    )
    if head.returncode != 0:
        return {}
    remote = git_lines("branch", "-r", "--list", "origin/main")
    hashes = git_lines("log", "--format=%H", "origin/main..HEAD" if remote else "HEAD")
    return {
        f"commit {sha[:7]}": "\n".join(git_lines("log", "-1", "--format=%B", sha))
        for sha in hashes
    }


def main() -> int:
    failed = []
    for name, command in STEPS:
        print(f"== {name}", flush=True)
        if subprocess.run(command).returncode != 0:
            failed.append(name)
    print("== attribution", flush=True)
    for problem in attribution_problems({**files_to_commit(), **unpushed_messages()}):
        print(problem)
        failed.append("attribution")
    if failed:
        print("FAILED: " + ", ".join(dict.fromkeys(failed)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
