"""Reading a web page as a visitor reads it, by its host's rules. Pages invented."""

import pytest

from petroknowledge import webpage

PAGE = """<html><head><title>Who We Are | Acme</title></head><body>
<div class="nav-bar"><a href="/">Home</a> <a href="/investors">Investors</a></div>
<header>Site header</header>
<div class="template-body">
  <div class="section-heading">Our Mandate</div>
  <p>We produce <b>crude oil</b> and gas.<br>Daily.</p>
  <form>Subscribe to updates</form><script>var inside = "script";</script>
  <p><strong>Key facts</strong></p>
  <ul><li>Founded 1977</li><li><strong>Abuja</strong></li></ul>
  <p><strong>Our mandate is to create an energy business that is:</strong></p>
  <dl><dt>Efficient</dt><dd>Lean operations.</dd></dl>
  <p class="h3-style">Styled heading</p>
  <p>Under the styled heading.</p>
  <img src="/plant.jpg" alt="The gas processing plant at night in Delta">
  <img src="/logo.png" alt="Logo">
  <img alt="no source at all here five words">
  <!-- a comment -->
  <h2>References</h2>
  <p>Source list</p>
  <h3>Books</h3>
  <p>A book</p>
  <h2>History</h2>
  <p>Founded in 1977.</p>
</div>
<footer>Contact us</footer>
<script>var x = "script text";</script>
</body></html>"""
RULES = {
    "main": ".template-body",
    "furniture": ".nav-bar",
    "headings": ".section-heading",
    "stop_headings": ["References"],
}


def test_a_page_is_read_by_its_hosts_rules() -> None:
    title, found = webpage.read_page(PAGE, RULES)

    assert title == "Who We Are | Acme"  # no <h1>: the document's title
    shown = [(e["kind"], e["text"], e["section"]) for e in found]
    assert shown == [
        ("heading", "Our Mandate", []),
        ("paragraph", "We produce crude oil and gas.", ["Our Mandate"]),
        ("paragraph", "Daily.", ["Our Mandate"]),
        ("heading", "Key facts", ["Our Mandate"]),  # a bold line of its own
        ("list_item", "Founded 1977", ["Our Mandate", "Key facts"]),
        ("list_item", "Abuja", ["Our Mandate", "Key facts"]),  # bold, but listed
        (
            "paragraph",
            "Our mandate is to create an energy business that is:",
            ["Our Mandate", "Key facts"],
        ),  # bold, but a lead-in
        ("heading", "Efficient", ["Our Mandate"]),  # a definition term
        ("paragraph", "Lean operations.", ["Our Mandate", "Efficient"]),
        ("heading", "Styled heading", ["Our Mandate"]),
        ("paragraph", "Under the styled heading.", ["Our Mandate", "Styled heading"]),
        (
            "figure",
            "The gas processing plant at night in Delta",
            ["Our Mandate", "Styled heading"],
        ),
        ("heading", "History", []),
        ("paragraph", "Founded in 1977.", ["History"]),
    ]
    assert found[11]["image"] == "/plant.jpg"


def test_without_rules_main_or_body_is_read_and_the_h1_is_the_title() -> None:
    html = "<body><main><h1>Annual Report</h1><p>Revenue rose.</p></main></body>"

    title, found = webpage.read_page(html, {})

    assert title == "Annual Report"
    assert found == [{"kind": "paragraph", "text": "Revenue rose.", "section": []}]
    _, every = webpage.read_page(
        '<body><img src="/a.png" alt="Logo"></body>', {}, every_image=True
    )
    assert every[0]["kind"] == "figure" and every[0]["image"] == "/a.png"


def test_a_missing_main_selector_falls_back_and_a_page_needs_a_body() -> None:
    _, found = webpage.read_page("<body><p>Text</p></body>", {"main": ".absent"})
    assert found[0]["text"] == "Text"

    with pytest.raises(ValueError, match="no <body>"):
        webpage.read_page("", {})


def test_title_block_lines_are_metadata() -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        '<div><p class="date">1 October</p><p>Body</p></div>', "html.parser"
    )
    root = soup.div
    assert root is not None
    lines = webpage.rendered_lines(root)

    found = webpage.elements(lines, root.select(".date"))

    assert [e["kind"] for e in found] == ["meta", "paragraph"]


def test_a_title_is_read_once_and_a_portal_s_form_can_hold_the_content() -> None:
    post = (
        "<html><head><title>Board Inaugurated | Acme</title></head><body><main>"
        "<p>MAY 1, 2026</p><p>Board Inaugurated</p><p>The board met.</p>"
        "</main></body></html>"
    )
    title, found = webpage.read_page(post, {})
    # No h1: the element showing the <title> (less the site's name) is the title.
    assert title == "Board Inaugurated"
    assert [e["text"] for e in found] == ["MAY 1, 2026", "The board met."]
    wiki = (
        "<html><head><title>Gas Act - Wiki</title></head><body><h1>Gas Act</h1>"
        "<div id='c'><p>Gas Act</p><p>Passed in 2021.</p></div></body></html>"
    )
    title, found = webpage.read_page(wiki, {"main": "#c"})
    assert title == "Gas Act" and len(found) == 2  # an h1 outside the content
    plain = "<html><head><title>Plain</title></head><body><p>Text</p></body></html>"
    assert webpage.read_page(plain, {})[0] == "Plain"  # nothing shows it
    portal = (
        "<body><main><form><h2>About</h2><p>We sell fuel.</p>"
        "<input value='x'></form></main></body>"
    )
    assert webpage.read_page(portal, {})[1] == []  # a form is furniture
    _, found = webpage.read_page(portal, {"forms_hold_content": True})
    assert [e["text"] for e in found] == ["About", "We sell fuel."]
