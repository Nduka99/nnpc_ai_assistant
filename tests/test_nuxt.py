"""Reading a page's text from its Nuxt data. The payloads are invented, in devalue's
flat format: containers hold the indices of their values."""

import json

from petroknowledge import nuxt


def page(values: object) -> str:
    """A page whose only text is in its Nuxt data."""
    data = json.dumps(values)
    return (
        f'<html><body><div id="__nuxt"></div><script type="application/json" '
        f'id="__NUXT_DATA__" data-ssr="true">{data}</script></body></html>'
    )


def test_devalue_wrappers_references_and_constants_are_rebuilt() -> None:
    values = [
        ["ShallowReactive", 1],
        {"a": 2, "b": 3, "c": -1, "d": 4, "e": 5, "f": 6, "g": 2},
        ["Reactive", 7],
        ["EmptyRef"],
        ["Set", 8, 8],
        ["Date", "2026-10-10T00:00:00.000Z"],
        [8, 7],
        {"x": 8},
        "text",
    ]

    assert nuxt.hydrate(values) == {
        "a": {"x": "text"},
        "b": None,
        "c": None,
        "d": ["text", "text"],
        "e": None,
        "f": ["text", {"x": "text"}],
        "g": {"x": "text"},  # a shared value is built once
    }


def test_malformed_data_is_none() -> None:
    assert nuxt.hydrate([{"data": "not an index"}]) is None
    assert nuxt.hydrate([{"data": 5}]) is None


PROFILE = [
    ["ShallowReactive", 1],  # 0
    {"data": 2, "state": 3},
    {"about-page": 4},
    {"menu": 5},  # site-wide state: never read
    {
        "description": 6,
        "title": 7,
        "createdAt": 8,
        "cta_text": 9,
        "hero_image": 10,
        "strategy": 12,
        "leaders": 16,
        "body": 20,
        "blocks": 21,
        "count": 29,
        "visible": 30,
        "nothing": -1,
    },
    "Home Investors Careers",  # 5
    "We produce crude oil and gas.",
    "Who We Are",
    "2026-07-21T11:55:23.348Z",
    "Visit Page",
    {"name": 11, "mime": 11, "url": 11},  # 10: a picture record
    "egina.jpg",
    {"title": 13, "inner_title": 14, "text": 15, "again": 15},
    "Our Strategy",
    "Growing energy delivery",
    "We will expand gas processing.",  # 15
    [17, 18],
    {"name": 19, "position": 15},
    {"name": 19, "position": 15},
    "Ada Obi",
    "<h2>Mandate</h2><p>Deliver energy.</p><ul><li>Efficient</li></ul>",  # 20
    [22, 25, 27],
    {"type": 31, "level": 32, "children": 33},
    {"type": 34, "text": 35},
    "unused",
    {"type": 36, "children": 37},  # 25
    {"type": 34, "text": 38},
    {"type": 39, "children": 40},
    {"type": 34, "text": 41},
    7,
    True,  # 30
    "heading",
    2,
    [23],
    "text",
    "Values",  # 35
    "paragraph",
    [26],
    "Integrity first.",
    "list-item",
    [28],  # 40
    "Excellence",
]


def test_a_pages_own_content_is_read_with_its_headings() -> None:
    found = nuxt.page_elements(page(PROFILE))

    shown = [(e["kind"], e["text"], e["section"], e.get("level")) for e in found]
    who = ["Who We Are"]
    assert shown == [
        ("heading", "Who We Are", [], 1),  # the title leads, though stored second
        ("paragraph", "We produce crude oil and gas.", who, None),
        ("heading", "Our Strategy", who, 2),
        ("heading", "Growing energy delivery", [*who, "Our Strategy"], 3),
        (
            "paragraph",
            "We will expand gas processing.",
            [*who, "Our Strategy", "Growing energy delivery"],
            None,
        ),
        ("heading", "Ada Obi", who, 2),
        ("paragraph", "We will expand gas processing.", [*who, "Ada Obi"], None),
        ("heading", "Ada Obi", who, 2),  # the repeated record's text is not repeated
        ("heading", "Mandate", who, 2),
        ("paragraph", "Deliver energy.", [*who, "Mandate"], None),
        ("list_item", "Efficient", [*who, "Mandate"], None),
        ("heading", "Values", who, 2),
        ("paragraph", "Integrity first.", [*who, "Values"], None),
        ("list_item", "Excellence", [*who, "Values"], None),
    ]
    assert nuxt.prose_words(found) == 22


def test_a_long_title_field_is_prose_and_the_first_heading_names_the_record() -> None:
    long_title = "A title field long enough to be a paragraph " * 4
    values = [
        {"data": 1},
        {"title": 2, "subtitle": 3, "inner": 4},
        long_title,
        "Short",
        {"heading": 3},
    ]

    found = nuxt.page_elements(page(values))

    assert [(e["kind"], e["text"]) for e in found] == [
        ("heading", "Short"),
        ("paragraph", " ".join(long_title.split())),
        ("heading", "Short"),
    ]


def test_a_lone_block_and_nested_lists_are_read() -> None:
    values = [
        {"data": 1},
        {"page": 2},
        {"intro": 3, "rows": 7},
        {"type": 4, "children": 5},  # a block on its own, not in a list
        "paragraph",
        [6, 11, 12],  # a child record, plain text and a number
        {"text": 8},
        [10],  # a list holding a list
        "Nested words",
        {"body": 8},
        [9],
        "and more",
        3,
    ]

    found = nuxt.page_elements(page(values))

    assert [(e["kind"], e["text"]) for e in found] == [
        ("paragraph", "Nested words and more"),
        ("paragraph", "Nested words"),
    ]


def test_pages_without_usable_data_give_nothing() -> None:
    assert nuxt.page_elements("<html><body><p>Plain page</p></body></html>") == []
    broken = '<script id="__NUXT_DATA__">{not json</script>'
    assert nuxt.page_elements(broken) == []
    assert nuxt.page_elements(page([])) == []
    assert nuxt.page_elements(page(["just text"])) == []
    assert nuxt.page_elements(page([{"state": 1}, {}])) == []
    assert nuxt.prose_words([]) == 0


def test_the_pages_own_date_is_the_shallowest_date_field() -> None:
    values = [
        {"data": 1},
        {"post": 2},
        {"related": 3, "post_date": 6, "image": 7},
        [4],
        {"post_date": 5},
        "2026-09-29",
        "2026-10-09",
        {"caption_date": 8, "publishedAt": 6},
        "not a date",
    ]

    assert nuxt.page_date(page(values)) == "2026-10-09"
    assert nuxt.page_date(page([{"data": 1}, {"date": 2}, "soon"])) == ""
    assert nuxt.page_date("<p>no data</p>") == ""


POST = [
    {"data": 1},
    {"insights-post": 2},
    {"singleBlogPost": 3, "otherPosts": 4},
    {"title": 5, "content": 6, "post_date": 7, "slug": 8, "excerpt": 9,
     "publishedOn": 10, "pdf_link": 11, "post_type": 12, "post_category": 13},
    {"data": 14, "meta": 15},
    "Monthly Report Summary - July 2025",
    "<p>NNPC released its monthly report.</p>",
    "2025-08-21",
    "monthly-report-july-2025",
    "Production and revenue figures.",
    "2025-08-21T17:24:06.537Z",
    "#",
    {"type": 16, "documentId": 25},
    {"category": 17, "documentId": 25},
    [18],
    {"pagination": 19},
    "News",
    "Company News",
    {"title": 20, "content": 21, "post_date": 22, "slug": 23, "excerpt": 21,
     "documentId": 25},
    {"page": 24},
    "Discount Offers Relief",
    "<p>Another post's full text.</p>",
    "2026-10-09",
    "discount-offers-relief",
    1,
    "kld0xe22tqcu5141hw81xfcf",
]  # fmt: skip


def test_a_post_is_read_without_related_posts_timestamps_or_labels() -> None:
    found = nuxt.page_elements(page(POST))

    assert [(e["kind"], e["text"]) for e in found] == [
        ("heading", "Monthly Report Summary - July 2025"),
        ("paragraph", "NNPC released its monthly report."),
        ("paragraph", "Production and revenue figures."),
    ]


def test_a_list_of_alike_page_components_is_the_pages_own_content() -> None:
    values = [
        {"data": 1},
        {"page": 2},
        {"intro": 3, "scopes": 7},
        {"title": 4, "description": 5, "value": 6},
        "Emissions",
        "Our footprint by scope.",
        "39.454",
        [8],
        {"title": 9, "description": 10, "value": 6},
        "Scope 1",
        "Direct emissions from operations.",
    ]

    found = nuxt.page_elements(page(values))

    assert [e["text"] for e in found] == [
        "Emissions",
        "Our footprint by scope.",
        "39.454",
        "Scope 1",
        "Direct emissions from operations.",
        "39.454",
    ]


# A post with its cover picture (and a resized copy) and an attached PDF, followed by
# another post with its own picture.
FILES = [
    {"data": 1},
    {"post": 2},
    {"singleBlogPost": 3, "otherPosts": 4},
    {"title": 5, "slug": 6, "post_date": 7, "post_image": 8, "upload_pdf_file": 9},
    {"data": 10},
    "Expression of Interest",
    "eoi",
    "2025-03-01",
    {"name": 11, "url": 12, "mime": 13, "formats": 14},
    {"name": 15, "url": 16, "mime": 17},
    [18],
    "flyer.png",
    "/uploads/flyer_ab12.png",
    "image/png",
    {"thumbnail": 19},
    "eoi.pdf",
    "/uploads/eoi_cd34.pdf",
    "application/pdf",
    {"title": 20, "slug": 21, "post_date": 22, "post_image": 23, "documentId": 24},
    {"url": 25, "mime": 13},
    "Other post",
    "other",
    "2025-02-01",
    {"name": 26, "url": 27, "mime": 13},
    "abc",
    "/uploads/thumbnail_flyer_ab12.png",
    "logo.jpg",
    "/uploads/logo.jpg",
]  # fmt: skip


def test_a_pages_own_uploads_are_found_at_their_original_size() -> None:
    assert nuxt.page_files(page(FILES)) == [
        "/uploads/flyer_ab12.png",
        "/uploads/eoi_cd34.pdf",
    ]
    gallery = [{"data": 1}, {"gallery": 2}, [3], {"url": 4, "mime": 5},
               "/uploads/g.jpg", "image/jpeg"]  # fmt: skip
    assert nuxt.page_files(page(gallery)) == ["/uploads/g.jpg"]
    assert nuxt.page_files("<html></html>") == []


def test_a_pages_own_media_come_once_each_with_their_alt_text() -> None:
    assert nuxt.page_media(page(FILES)) == [
        {"url": "/uploads/flyer_ab12.png", "alt": ""},
        {"url": "/uploads/eoi_cd34.pdf", "alt": ""},
    ]
    gallery = [{"data": 1}, {"gallery": 2}, [3, 7, 9],
               {"url": 4, "mime": 5, "alternativeText": 6}, "/uploads/g.jpg",
               "image/jpeg", "A crude oil tanker at Bonny terminal",
               {"url": 4, "mime": 5}, None,
               {"url": 10, "mime": 5, "caption": 11}, "/uploads/h.jpg",
               "Board members"]  # fmt: skip
    assert nuxt.page_media(page(gallery)) == [
        {"url": "/uploads/g.jpg", "alt": "A crude oil tanker at Bonny terminal"},
        {"url": "/uploads/h.jpg", "alt": "Board members"},
    ]
    assert nuxt.page_media("<html></html>") == []


def test_a_rich_text_field_s_style_is_not_text() -> None:
    html = "<style>.tree-nav{color:black}</style><p>Disclosures 2023</p>"

    found = nuxt.markup_elements(html, ["Data"])

    assert [element["text"] for element in found] == ["Disclosures 2023"]
