"""Scanned pages read by a vision model. Every page, answer and reading is invented;
no model runs."""

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pypdfium2
import pytest
from PIL import Image

from petroknowledge import cli, config, llm, regions, transcribe
from petroknowledge.elements import Element
from petroknowledge.regions import Region

FINGERPRINT = {"vision_model": "GLM-OCR-f16.gguf", "prompt": "Text Recognition:"}


def reading(sha256: str, flags: list[list[str]]) -> dict[str, Any]:
    pages = [{"page": n, "flags": f} for n, f in enumerate(flags, 1)]
    return {"sha256": sha256, "urls": [f"https://x/{sha256[:4]}.pdf"],
            "validation": {"pages": pages}}  # fmt: skip


def test_only_pages_with_little_or_no_text_layer_are_transcribed() -> None:
    flags = [[], ["little or no text"], ["little or no text", "poor confidence"],
             ["low coverage"]]  # fmt: skip

    assert transcribe.scanned_pages(reading("a" * 64, flags)) == [2, 3]


def test_a_markdown_answer_reads_as_headings_lists_tables_and_paragraphs() -> None:
    prose = (
        "Revenue rose to $5 million and $6 million in 2025 because crude output and "
        "gas sales both increased over the year."
    )
    answer = f"""# Annual Report
## Pension
Mortality rate

| Age | Deaths $\\%$ | Rate |
| :--- | ---: | --- |
| 25 | 7 |
| 30 | 9 | 0.1 |
- Valued at  $40^{{\\circ}}$C
{prose}
| Only | Header |
## Notes
"""
    found = transcribe.markdown_elements(answer, 4, first_table=3)

    pension = ("Annual Report", "Pension")
    grid = (("Age", "Deaths %", "Rate"), ("25", "7", ""), ("30", "9", "0.1"))
    assert found == [
        Element(4, "heading", "Annual Report", ()),
        Element(4, "heading", "Pension", ("Annual Report",)),
        Element(4, "paragraph", "Mortality rate", pension),
        Element(4, "table", "Mortality rate\nAge | Deaths % | Rate\n25 | 7 | \n"
                "30 | 9 | 0.1", pension, table=3, grid=grid),
        Element(4, "table_header", "Age | Deaths % | Rate", pension, table=3),
        Element(4, "table_row", "Mortality rate › 25 — Deaths %: 7", pension, table=3,
                row=0, cells=(("Age", "25"), ("Deaths %", "7"), ("Rate", ""))),
        Element(4, "table_row", "Mortality rate › 30 — Deaths %: 9; Rate: 0.1",
                pension, table=3, row=1,
                cells=(("Age", "30"), ("Deaths %", "9"), ("Rate", "0.1"))),
        Element(4, "list", "Valued at 40°C", pension),
        Element(4, "paragraph", prose, pension),  # amounts' dollar signs kept
        Element(4, "table", "Only | Header", pension, table=4,
                grid=(("Only", "Header"),)),  # long prose above is not a caption
        Element(4, "table_header", "Only | Header", pension, table=4),
        Element(4, "heading", "Notes", ("Annual Report",)),
    ]  # fmt: skip


def test_a_cell_over_several_lines_stays_in_its_row_and_a_rule_is_no_table() -> None:
    answer = """|---|---|

Table 1
| S/N | Parameter | Method |
| 1. | Distillation:
   a) Recovery at 200°C
   b) Final boiling point | ASTM D 86 |
| 2. | Flash point | ASTM D 56 |"""

    found = transcribe.markdown_elements(answer, 2)

    assert [(e.kind, e.text, e.table) for e in found] == [
        ("paragraph", "Table 1", None),
        ("table", "Table 1\nS/N | Parameter | Method\n1. | Distillation: a) "
         "Recovery at 200°C b) Final boiling point | ASTM D 86\n"
         "2. | Flash point | ASTM D 56", 1),
        ("table_header", "S/N | Parameter | Method", 1),
        ("table_row", "Table 1 › 1. — Parameter: Distillation: a) Recovery at 200°C "
         "b) Final boiling point; Method: ASTM D 86", 1),
        ("table_row", "Table 1 › 2. — Parameter: Flash point; Method: ASTM D 56", 1),
    ]  # fmt: skip


def test_saved_answers_become_elements_with_tables_numbered_in_page_order() -> None:
    transcript = {"pages": {"10": {"answer": "| a |\n| 1 |"},
                            "9": {"answer": "| b |\n| 2 |"}}}  # fmt: skip

    found = transcribe.transcript_elements(transcript)

    tables = [(e.page, e.table, e.text) for e in found if e.kind == "table"]
    assert tables == [(9, 1, "b\n2"), (10, 2, "a\n1")]


def test_a_page_is_read_whole_with_its_furniture_and_pictures_painted_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scan.pdf"
    document = pypdfium2.PdfDocument.new()
    document.new_page(72, 36)  # one inch by half an inch, in points
    document.save(path)
    document.close()
    drawn: list[int] = []
    original = transcribe.page_image

    def page_image(source: Path, number: int, dpi: int = 200) -> Image.Image:
        drawn.append(dpi)
        found = original(source, number, dpi)
        return Image.new("RGB", found.size, "black")  # a page printed all over

    found = [
        Region("page_header", 0.9, (0, 0, 72, 4)),
        Region("picture", 0.9, (50, 10, 70, 30)),
        Region("table", 0.9, (4, 12, 40, 30)),  # read with the page
    ]

    def find(image: Image.Image, processor: str, model: str) -> list[Region]:
        assert image.size == (72, 36) and (processor, model) == ("processor", "model")
        return found

    monkeypatch.setattr(transcribe, "page_image", page_image)
    monkeypatch.setattr(transcribe, "layout", lambda: ("processor", "model"))
    monkeypatch.setattr(regions, "find", find)
    sent: dict[str, Any] = {}
    replies = iter([
        httpx.Response(200, json={"choices": [
            {"message": {"content": "Page text"}, "finish_reason": "stop"}]}),
        httpx.Response(200, json={"choices": [
            {"message": {"content": "1 1 1 1"}, "finish_reason": "length"}]}),
        httpx.Response(500),
    ])  # fmt: skip

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.update(url=url, **kwargs)
        reply = next(replies)
        reply.request = httpx.Request("POST", url)
        return reply

    monkeypatch.setattr(llm.CLIENT, "post", post)
    monkeypatch.setattr(llm, "auth", lambda: {})

    assert transcribe.read_page(path, 1, "http://vlm") == "Page text"
    assert drawn == [72, 200]
    assert sent["url"] == "http://vlm/v1/chat/completions"
    content = sent["json"]["messages"][0]["content"]
    assert content[1] == {"type": "text", "text": "Text Recognition:"}
    assert sent["json"]["top_k"] == 1 and sent["json"]["temperature"] == 0.0
    picture = content[0]["image_url"]["url"].removeprefix("data:image/png;base64,")
    image = Image.open(BytesIO(base64.b64decode(picture))).convert("RGB")
    assert image.size == (200, 100)
    white, black = (255, 255, 255), (0, 0, 0)
    assert image.getpixel((100, 5)) == white  # the page header
    assert image.getpixel((166, 55)) == white  # the picture
    assert image.getpixel((60, 55)) == black  # the table stays
    assert transcribe.read_page(path, 1, "http://vlm") is None  # a repetition loop
    with pytest.raises(llm.ModelServerError, match="500"):
        transcribe.read_page(path, 1, "http://vlm")


def test_the_layout_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[int] = []

    def load() -> tuple[str, str]:
        loads.append(1)
        return "processor", "model"

    transcribe.layout.cache_clear()
    monkeypatch.setattr(regions, "load", load)

    assert transcribe.layout() == transcribe.layout() == ("processor", "model")
    assert loads == [1]
    transcribe.layout.cache_clear()


def test_a_table_written_in_html_keeps_each_spanning_cell_in_every_position() -> None:
    answer = """## Royalties
<table><tr><th></th><th colspan="2">31 December 2022</th></tr>
<tr><td>Additions</td><td>677,231</td><td>520,815</td></tr></table>
<table></table>
Closing note
<table><tr><td>a $10\\%$</td></tr>"""

    found = transcribe.markdown_elements(answer, 3)

    tables = [(e.kind, e.table, e.grid, e.section) for e in found if e.table]
    assert tables[0] == ("table", 1, (("", "31 December 2022", "31 December 2022"),
                                      ("Additions", "677,231", "520,815")),
                         ("Royalties",))  # fmt: skip
    assert found[2].kind == "table_header"
    assert found[3].cells == (("", "Additions"), ("31 December 2022", "677,231"),
                              ("31 December 2022", "520,815"))  # fmt: skip
    assert found[4] == Element(3, "paragraph", "Closing note", ("Royalties",))
    assert tables[-2][:3] == ("table", 2, (("a 10%",),))  # never closed: still kept


def test_a_fingerprint_names_the_model_prompt_sampling_and_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get(url: str, **kwargs: Any) -> httpx.Response:
        assert url == "http://vlm/v1/models"
        body = {"data": [{"id": "GLM-OCR-f16.gguf"}]}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(llm, "auth", lambda: {})

    code = Path(regions.__file__).read_bytes()
    assert transcribe.fingerprint("http://vlm") == {
        "vision_model": "GLM-OCR-f16.gguf",
        "layout_model": "docling-project/docling-layout-heron@"
                        "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
        "regions_sha256": hashlib.sha256(code).hexdigest(),
        "masked": ["page_header", "page_footer", "picture"],
        "prompt": "Text Recognition:",
        "max_tokens": 8192,
        "sampling": {"temperature": 0.0, "top_p": 0.00001, "top_k": 1,
                     "repeat_penalty": 1.1},
        "dpi": 200,
        "layout_dpi": 72,
    }  # fmt: skip


def test_a_run_transcribes_each_scanned_page_once_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = tmp_path / "elements"
    readings.mkdir()
    scan, typed = "a" * 64, "b" * 64
    scanned = ["little or no text"]
    for found in (reading(scan, [[], scanned, scanned, scanned]), reading(typed, [[]])):
        (readings / f"{found['sha256']}.json").write_text(json.dumps(found))
    (readings / "report.json").write_text("{}")
    (readings / f"{scan}.part.json").write_text("{}")
    monkeypatch.setattr(config, "FILE_STORE", tmp_path / "files")
    monkeypatch.setattr(transcribe, "fingerprint", lambda url: FINGERPRINT)
    asked: list[tuple[Path, int]] = []
    unfinished = {3}

    def read(path: Path, number: int, url: str) -> str | None:
        asked.append((path, number))
        if number == 4:
            raise llm.ModelServerError("timed out")
        return None if number in unfinished else f"page {number}"

    out = tmp_path / "transcripts"
    first = transcribe.transcribe_all("http://vlm", readings, out, read)

    assert asked == [(tmp_path / "files" / scan, 2), (tmp_path / "files" / scan, 3),
                     (tmp_path / "files" / scan, 4)]  # fmt: skip
    assert first == {"documents": 1, "pages": 1, "skipped": 0, "failed": [
        {"url": "https://x/aaaa.pdf", "page": 3, "error": "answer not finished"},
        {"url": "https://x/aaaa.pdf", "page": 4, "error": "timed out"},
    ]}  # fmt: skip
    saved = json.loads((out / f"{scan}.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == FINGERPRINT
    assert saved["pages"]["2"]["answer"] == "page 2"
    assert not list(out.glob("*.part"))

    unfinished.clear()
    asked.clear()
    second = transcribe.transcribe_all("http://vlm", readings, out, read)

    assert [number for _, number in asked] == [3, 4]
    assert (second["pages"], second["skipped"], len(second["failed"])) == (1, 1, 1)

    asked.clear()
    monkeypatch.setattr(transcribe, "fingerprint", lambda url: {"vision_model": "new"})
    transcribe.transcribe_all("http://vlm", readings, out, read)

    assert [number for _, number in asked] == [2, 3, 4]  # another model reads again

    asked.clear()
    elsewhere = tmp_path / "only"
    transcribe.transcribe_all("http://vlm", readings, elsewhere, read, {(scan, 3)})

    assert [number for _, number in asked] == [3]  # only the listed page


def test_the_command_line_transcribes_and_prints_the_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report: dict[str, Any] = {
        "documents": 2,
        "pages": 5,
        "skipped": 1,
        "failed": [
            {"url": "https://x/scan.pdf", "page": 7, "error": "answer not finished"}
        ],
    }
    calls: list[tuple[str, Path, Path, Any]] = []

    def transcribe_all(
        url: str, readings: Path, out: Path, only: Any = None
    ) -> dict[str, Any]:
        calls.append((url, readings, out, only))
        return report

    monkeypatch.setattr(transcribe, "transcribe_all", transcribe_all)
    monkeypatch.setattr(llm, "healthy", lambda url: url == "http://up")

    assert cli.main(["transcribe", "--vlm", "http://down"]) == 1
    assert "no vision model ready" in capsys.readouterr().err
    assert cli.main(["transcribe", "--vlm", "http://up"]) == 1
    printed = capsys.readouterr().out
    assert calls == [("http://up", config.READINGS, config.TRANSCRIPTS, None)]
    assert "2 documents with scanned pages: 5 pages transcribed, 1 already" in printed
    assert "failed scan.pdf p7: answer not finished" in printed
    report["failed"] = []
    listed = tmp_path / "truth.json"
    listed.write_text(json.dumps({"pages": [{"sha256": "ab", "page": 3}]}))
    out = tmp_path / "out"
    assert cli.main(["transcribe", "--vlm", "http://up", "--pages", str(listed),
                     "--out", str(out)]) == 0  # fmt: skip
    assert calls[-1] == ("http://up", config.READINGS, out, {("ab", 3)})
