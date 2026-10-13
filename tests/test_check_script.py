"""The Rule 3 scan in scripts/check.py: method records may name AI tools, nothing
else may, and no file or commit message may carry an attribution trailer."""

from scripts.check import attribution_problems


def test_method_records_may_name_tools_but_not_attribute() -> None:
    texts = {
        "README.md": "Claude and Codex were coding tools; a Gemini chat judged.",
        "src/petroknowledge/store.py": "cursor = conn.cursor()  # a database Cursor",
    }

    assert attribution_problems(texts) == []


def test_attribution_and_stray_names_are_reported() -> None:
    texts = {
        "README.md": "Co-Authored-By: Claude <noreply@anthropic.com>",
        "commit abc1234": "Add store\n\nGenerated with [Claude Code](https://x)",
        "src/petroknowledge/api.py": "# written with copilot",
    }

    assert attribution_problems(texts) == [
        "README.md: AI attribution",
        "commit abc1234: AI attribution",
        "src/petroknowledge/api.py: names an AI tool outside a method section",
    ]


def test_the_scanner_and_its_test_are_not_scanned() -> None:
    texts = {"scripts/check.py": 'ATTRIBUTION = re.compile(r"co-authored-by")'}

    assert attribution_problems(texts) == []
