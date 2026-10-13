"""K12's gate: scanned pages drawn by seed, their transcriptions scored against truth
written from the page images. Every document, page and text here is invented."""

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import httpx
import pypdfium2
import pytest
from PIL import Image

from evaluation import __main__ as cli
from evaluation import scans

A, B, C = "a" * 64, "b" * 64, "c" * 64
SCANNED = ["little or no text"]


def reading(sha256: str, flags: list[list[str]]) -> dict[str, Any]:
    pages = [{"page": n, "flags": f} for n, f in enumerate(flags, 1)]
    return {"sha256": sha256, "urls": [f"https://x/{sha256[:4]}.pdf?v=1"],
            "validation": {"pages": pages}}  # fmt: skip


READINGS = {
    A: reading(A, [SCANNED, [], SCANNED, SCANNED]),
    B: reading(B, [SCANNED, SCANNED]),
    C: reading(C, [SCANNED, SCANNED, SCANNED]),
}
STRATA = {
    "strata": {"reports": [A, B], "agreements": [C]},
    "exclude": [{"sha256": A, "page": 3}],
}


def test_candidates_are_each_stratum_s_unseen_scanned_pages() -> None:
    found = scans.candidates(STRATA, READINGS)

    assert [(p["sha256"][0], p["page"]) for p in found["reports"]] == [
        ("a", 1), ("a", 4), ("b", 1), ("b", 2),
    ]  # fmt: skip
    assert found["agreements"][0] == {
        "stratum": "agreements", "sha256": C, "page": 1, "url": "https://x/cccc.pdf?v=1"
    }  # fmt: skip


def test_a_draw_is_seeded_per_stratum_with_an_ordered_reserve() -> None:
    first = scans.draw(scans.candidates(STRATA, READINGS), 7, 1, {"reports": 2})
    again = scans.draw(scans.candidates(STRATA, READINGS), 7, 1, {"reports": 2})

    assert first == again
    assert [p["stratum"] for p in first["pages"]] == ["agreements", "reports"]
    assert [p["stratum"] for p in first["reserve"]] == ["reports", "reports"]
    assert [p["number"] for p in first["pages"] + first["reserve"]] == [1, 2, 3, 4]
    reports = {(p["sha256"], p["page"]) for p in first["pages"][1:] + first["reserve"]}
    assert len(reports) == 3  # no page drawn twice


def test_a_page_is_saved_as_two_overlapping_halves_at_200_dpi(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    document = pypdfium2.PdfDocument.new()
    document.new_page(72, 144)  # one inch by two
    document.save(source)
    document.close()

    saved = scans.save_images(source, 1, tmp_path / "p01")

    assert [path.name for path in saved] == ["p01-a.png", "p01-b.png"]
    assert [Image.open(path).size for path in saved] == [(200, 220), (200, 220)]


def test_a_transcription_is_scored_by_its_words_and_numbers() -> None:
    page: dict[str, Any] = {
        "text": "Statement of financial position As at 31 December 2022",
        "tables": [[["", "Note", "₦'million"], ["Revenue", "5", "1,234,567"]]],
    }
    answer = (
        "## Statement of **financial** position\nAs at 31 December 2022\n\n"
        "| | Note | ₦'million |\n| :--- | --- | ---: |\n"
        "| Revenue | 6 | 1,234,567 |\n| $10\\%$ |"
    )

    scored = scans.score_page(page, answer)

    assert scored == {
        "recall": (13, 14),  # the note number 5 was read as 6
        "precision": (13, 15),  # and 10% is not on the page
        "number recall": (3, 4),
        "number precision": (3, 5),
    }
    assert scans.unmarked("<b>x</b> | $y$ US$").split() == ["x", "y", "US"]
    shares = scans.pooled([scored, {m: (0, 0) for m in scans.GATE}])
    assert shares["recall"] == 13 / 14
    assert scans.pooled([])["recall"] == 0.0
    assert scans.passed(shares) == dict.fromkeys(scans.GATE, False)
    perfect = scans.score_page(page, " ".join([page["text"], "Note ₦'million",
                                                "Revenue 5 1,234,567"]))  # fmt: skip
    assert all(scans.passed(scans.pooled([perfect])).values())


def write_set(root: Path) -> Path:
    folder = root / "scans"
    folder.mkdir(parents=True)
    (folder / "strata.json").write_text(json.dumps(STRATA))
    return folder


def test_the_command_line_draws_pages_once_and_saves_their_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = write_set(tmp_path / "eval")
    readings, store = tmp_path / "elements", tmp_path / "files"
    readings.mkdir()
    store.mkdir()
    for sha256, found in READINGS.items():
        (readings / f"{sha256}.json").write_text(json.dumps(found))
        document = pypdfium2.PdfDocument.new()
        for _ in found["validation"]["pages"]:
            document.new_page(72, 72)
        document.save(store / sha256)
        document.close()
    args = ["--root", str(tmp_path / "eval"), "scan-sample", "scans", "--seed", "3",
            "--per-stratum", "1", "--reserve", "reports=1",
            "--readings", str(readings), "--store", str(store)]  # fmt: skip

    assert cli.main(args) == 0
    drawn = json.loads((folder / "pages.json").read_text(encoding="utf-8"))
    assert (
        drawn["seed"] == 3 and len(drawn["pages"]) == 2 and len(drawn["reserve"]) == 1
    )
    assert len(list((folder / "images").glob("*.png"))) == 6
    assert "2 pages, 1 reserve" in capsys.readouterr().out
    assert cli.main(args) == 1
    assert "drawn once" in capsys.readouterr().err


def test_the_command_line_scores_a_registered_set_and_reports_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    folder = write_set(root)
    truth = [{"number": 1, "sha256": A, "page": 1, "text": "Profit rose 12%",
              "tables": []}]  # fmt: skip
    (folder / "truth.json").write_text(json.dumps({"pages": truth}))
    common = ["--root", str(root), "--registry", str(registry)]
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    score = [*common, "scan-score", "scans", "--transcripts", str(transcripts),
             "--out", str(tmp_path / "score.json")]  # fmt: skip

    assert cli.main([*common, "scan-score", "scans", "--out", "x.json"]) == 1
    assert "not registered" in capsys.readouterr().err
    assert cli.main([*common, "register", "scans", "--description", "demo"]) == 0
    assert cli.main([*common, "verify"]) == 0
    assert "1 sets checked, 0 problems" in capsys.readouterr().out
    assert cli.main(score) == 1
    assert f"not transcribed yet: {A[:12]} p1" in capsys.readouterr().err

    saved = {"pages": {"1": {"answer": "Profit rose 12%"}}}
    (transcripts / f"{A}.json").write_text(json.dumps(saved))
    assert cli.main(score) == 0
    assert "recall: 1.000 (bar 0.95)" in capsys.readouterr().out
    report = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    assert all(report["gate"].values()) and report["pages"]["1"]["recall"] == [3, 3]

    saved["pages"]["1"]["answer"] = "Profit rose 17%"
    (transcripts / f"{A}.json").write_text(json.dumps(saved))
    assert cli.main(score) == 1
    assert capsys.readouterr().out.endswith("FAIL\n")
    (folder / "truth.json").write_text("{}")
    assert cli.main([*common, "verify", "scans"]) == 1


def test_the_module_runs_as_a_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = tmp_path / "sets.json"
    monkeypatch.setattr(sys, "argv", ["evaluation", "--registry", str(registry),
                                      "verify"])  # fmt: skip
    monkeypatch.delitem(sys.modules, "evaluation.__main__", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("evaluation", run_name="__main__")
    assert stopped.value.code == 0


def test_an_arm_asks_with_the_published_settings_and_drops_unfinished_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}
    replies = iter(["stop", "length"])

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.update(url=url, **kwargs)
        choice = {"message": {"content": "Page text"}, "finish_reason": next(replies)}
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"choices": [choice]}, request=request)

    monkeypatch.setattr(httpx, "post", post)
    image = Image.new("RGB", (4, 4))

    found = scans.ask(image, "http://vlm", "Table Recognition:")

    assert (found["answer"], found["finished"]) == ("Page text", True)
    assert sent["url"] == "http://vlm/v1/chat/completions"
    assert sent["json"]["messages"][0]["content"][1]["text"] == "Table Recognition:"
    assert sent["json"]["top_k"] == 1 and sent["json"]["repeat_penalty"] == 1.1
    assert scans.ask(image, "http://vlm", "x")["answer"] == ""


def test_a_dash_between_numbers_splits_them_however_it_is_spaced() -> None:
    assert scans.unmarked("6- 10, 5 - 7, 2022 - 1,686,102, non-current") == (
        "6 10, 5 7, 2022 1,686,102, non-current"
    )


def test_the_command_line_runs_an_arm_and_scores_its_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    folder = write_set(root)
    truth = [{"number": 1, "sha256": A, "page": 1, "text": "Profit rose 12%",
              "tables": []}]  # fmt: skip
    (folder / "truth.json").write_text(json.dumps({"pages": truth}))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "scans"]) == 0
    monkeypatch.setattr(scans, "page_image", lambda source, page: (source, page))
    answer = {"answer": "Profit rose 12%", "finished": False, "seconds": 1.0}
    monkeypatch.setattr(scans, "ask", lambda image, url, prompt: answer)
    run = tmp_path / "run.json"
    arm = ["scan-transcribe", "scans", "--vlm", "http://vlm", "--prompt", "Text",
           "--out", str(run), "--store", str(tmp_path)]  # fmt: skip

    assert cli.main([*common, *arm]) == 0
    assert "1 pages read, 1 unfinished" in capsys.readouterr().out
    assert json.loads(run.read_text())["prompt"] == "Text"
    score = [*common, "scan-score", "scans", "--out", str(tmp_path / "s.json")]
    assert cli.main([*score, "--run", str(run)]) == 0
    run.write_text(json.dumps({"pages": {}}))
    assert cli.main([*score, "--run", str(run)]) == 1
    assert "not transcribed yet" in capsys.readouterr().err
