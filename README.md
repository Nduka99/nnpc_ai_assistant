# PetroKnowledge

An independent evidence assistant over public oil and gas company information. It
starts with NNPC Limited, then the Dangote refinery and Nigeria LNG. The aim is the
same contract as the Lime Green assistant it extends: every claim shows a quotation
that code found in its source, and questions the sources cannot answer get a fixed
refusal.

**Independent.** This project is not affiliated with, endorsed by or connected to
NNPC Limited, Dangote Petroleum Refinery, Nigeria LNG or any other company it reads
about. It uses no company logos.

> **Status, October 2026.** Collection and reading are built, and every reader has
> been measured against truth judged from the sources themselves (see Evaluation).
> The index is being built: passages, the corpus builder and the Postgres schema are
> in place; storage, search, answers, the knowledge graph and the web app come next.

Each company's sources are in `companies/<id>/`.

## What is collected, and what is never published

Each company has a folder, `companies/<company>/`, with its `sources.json`; the news feeds they share are in `companies/news.json`. A company
without a `sources.json` is still being planned and is not collected. Two tiers:

- **Core sources** are read and indexed:
  - site pages;
  - the company's own posts (press releases, statements, report notices), each with
    its publication date;
  - annual reports and financial statements;
  - the regulations the company works under;
  - Wikipedia articles (CC BY-SA 4.0) and other openly licensed public data.
- **Linked sources** are third-party news items from RSS feeds. Only their metadata
  is kept (URL, title, description, date), and they are read live when an answer
  needs them.

Collection obeys each host's `robots.txt`, including its crawl-delay. It sends at
most one request a second per host, with an identifying user agent. It never follows
a redirect off the company's allowed hosts. Files larger than the company's limit
are skipped.

Collected content stays on the machine that collected it. `data/` is git-ignored
and this repository holds no company content. NNPC's terms of use, for example,
allow no copy beyond personal use without written permission. The system is run
as a private demonstration. Video is kept as links only (YouTube's terms), and social
media is not collected. This is not legal advice.

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). The same commands work
on Linux, macOS and Windows; the lockfile pins wheels for each. One command installs
the application, its tests and the `ingest` group (Docling, the layout-aware PDF
reader, with CPU-only PyTorch):

```sh
uv sync --all-groups --locked
```

The web page needs Node.js 24 and npm; one command installs its build and test tools
from the lockfile:

```sh
npm ci --prefix web
```

## Collect

Run every command from the repository root.

```sh
uv run petroknowledge acquire nnpc --measure           # size the core sources (HEAD requests)
uv run petroknowledge acquire nnpc                     # collect them
uv run petroknowledge acquire nnpc --linked --measure  # size the files those pages link
uv run petroknowledge acquire nnpc --linked            # collect them
uv run petroknowledge download-listed nnpc             # on the owner's instruction (below)
uv run petroknowledge add nnpc                         # store the inbox's files (below)
uv run petroknowledge links nnpc                       # keep NNPC's posts as links
uv run petroknowledge news                             # keep news items naming each company
uv run petroknowledge browse                           # readable names under data/browse/
uv run petroknowledge models                           # check models/ against models.json
```

Some hosts forbid crawlers while publishing documents for people to read. NNPC's
reports, policies, contracts and data files are served from hosts whose robots.txt
disallows everything, so the collector never fetches them. Two ways bring them in:

- **By hand.** A person opens those public links in a browser and saves the files
  into `data/inbox/<company>/` under the name at the end of each link.
- **On the owner's instruction.** `download-listed` fetches exactly the files named
  in `data/inbox/<company>-to-download.txt` into the inbox. It crawls nothing,
  identifies itself as the collector does and waits five seconds between files.

`add` stores each file with its original URL and how it came: "hand", or "download on
the owner's instruction (robots.txt disallows crawlers)". It refuses any file whose
name matches no link the company's sources list or its stored pages carry.

Each collected file is stored once in `data/files/`, named by its SHA-256.
`data/files/manifest.json` records its URL, company, kind, size, type, ETag,
Last-Modified and fetch time. A stopped run resumes, and a failed URL is retried next
time.

## Read

```sh
uv run petroknowledge read-pages nnpc   # every stored page into elements (data/pages/)
uv run petroknowledge read-documents    # every stored PDF and spreadsheet, each in its own process (data/elements/)
uv run petroknowledge transcribe --vlm http://127.0.0.1:8083   # scanned PDF pages, by GLM-OCR (data/transcripts/)
```

**Pages.** Each page is read the way a browser lays it out, keeping the headings
above each line. Sites differ in where their content sits, so each host's reading
rules are data in `companies/hosts.json`:
- the main content's selector;
- furniture to remove;
- elements that serve as headings;
- sections that are not content, such as a Wikipedia article's references.

Some sites build part of their text in the browser from data sent inside the page
(Nuxt's `__NUXT_DATA__`). NNPC's "Who we are" page has about 6,000 words there and
far fewer in its HTML. When a page's data holds more than twice the prose of its
HTML, the page is read from the data. Near-equal amounts are the same text, read
from the HTML a visitor sees. The data reader keeps to the page's own content. It
skips:
- related posts, meaning separate entries sharing most of a sibling's fields;
- bare dates and timestamps;
- symbol-only values;
- one-label category entries.

Rendering pages in a headless browser was tried first and rejected. It took about
95 requests and 30–54 seconds per NNPC page, its load event never fired, and the
text was already in the HTML collected.

**PDFs.** Docling reads layout and reading order, GLM-OCR can read tables again from
their images, and pdfium decides which words each page shows. Each file is read in
its own process, fingerprinted and validated per page, and the run's report flags
low-coverage pages.

**Scanned pages.** The PDF reader runs no OCR, so a page with almost no text layer
is flagged. NNPC publishes many such pages: annual reports, financial statements
and licence agreements scanned as pictures. `transcribe` draws each flagged page at
200 DPI and has GLM-OCR read it into Markdown: paragraphs, lists and tables.
- Only the model's answers are saved, with the model, prompt, token limit and
  resolution that made them. A run resumes page by page.
- An answer the model did not finish (a repetition loop at the token limit) is
  reported and tried again next run, never kept.
- The answers become the same elements PDF tables give. A short line directly above
  a table is its caption. Changing this conversion needs no new model run.
- Unlike a table read again, whose cells must be spelt from the PDF's own words, a
  transcription keeps the model's text: the page has none to check it against. No
  transcription passed its accuracy gate, so transcripts help search find a scanned
  page but are never quoted; an answer points to the page instead.

**Spreadsheets** (.xlsx with openpyxl, .xls with xlrd) are read into the same
elements PDF tables give: a whole table, its header, and one element per row with
every value paired with its column header. NNPC's production, lifting and gas data
come as spreadsheets.
- A multi-row header (a group label, the column names, a row of units) becomes one
  header per column, so a figure keeps its row label, column and unit in the text
  quoted.
- Merged cells repeat across the columns they cover.
- Blank margin columns are dropped.
- Numbers keep 15 significant digits and dates read as ISO dates.

The reading modules (`pdf`, `visibility`, `recovery`, `rendering`, `tables`,
`layout`, `documents`, `webpage`), the model-server client and telemetry are adopted
from the author's MIT-licensed Lime Green assistant. Comments that cite experiment
codes such as X42 or ADR 0027 refer to that project's records.

## Evaluation

Each reader is gated before indexing, with the Lime Green assistant's methods and
bars: values drawn by seed, truth written from page images or browser renderings
before any reading is looked at, every set registered by SHA-256 in
`evaluation/sets.json`, and a fix kept only if it holds on pages it was not derived
from. The sets themselves stay private with the corpus.

```sh
uv run python -m evaluation verify                      # every registered set's hashes
uv run python -m evaluation pdf-score SET --out FILE    # born-digital PDF tables
uv run python -m evaluation sheet-score SET --out FILE  # spreadsheet values
uv run --group evaluation python -m evaluation web-render SET   # pages as a visitor sees them
uv run python -m evaluation web-score SET --out FILE    # web pages
```

## Index database

The index lives in Postgres with pgvector (vector search) and pg_textsearch (BM25
keyword search), run with Docker Compose. Copy `deploy/.env.example` to `deploy/.env`
and set a password, then, from a Linux shell (WSL on Windows):

```sh
docker compose -f deploy/compose.yaml --profile dev up -d   # Postgres on 127.0.0.1:5433, migrations applied
```

Commands that use the database read `PETROKNOWLEDGE_DATABASE_URL` from `.env`
(git-ignored), passed with `uv run --env-file .env ...`.

An index version is built from the saved readings. It needs the embedding server
(Qwen3-Embedding-4B on llama.cpp, port 8084):

```sh
uv run --env-file .env petroknowledge index nnpc          # a new version, not yet served
uv run --env-file .env petroknowledge index nnpc --live   # ... and make it the one served
uv run --env-file .env petroknowledge index nnpc --read-roles  # read new role statements (chat model on 8080)
```

Every passage keeps its source, kind and search channel (the company's own content,
guidance such as regulators and Wikipedia, or pictures), and whether it may be quoted.
Scanned pages' transcripts and the text read in pictures are searched but never
quoted: neither reading passed its gate. A passage already embedded by an earlier
version keeps its vector, so a rebuild embeds only new text.

## Ask

```sh
uv run --env-file .env petroknowledge ask "Who is NNPC's Group Chief Executive Officer?"
```

The answer is a list of short claims, each backed by a quote checked word for word
against a quotable source. Each source is printed with its link and dates. A claim
whose quote, number, amount or regulation is not in its source is removed, never
repaired. The reader then sees which parts of the question were not answered.

- A request for investment, legal or safety advice gets fixed referral text.
- Pages that search found but that may not be quoted (scanned pages, words read in
  pictures) are listed for the reader to open.
- Every answer is stored for audit with the passages, prompts and index version used.

It needs three local model servers (llama.cpp):
- the embedder on port 8084;
- the reranker (bge-reranker-v2-m3) on 8082;
- the generator, Gemma 4 26B-A4B, on 8080.

## How to ask

These habits were measured on development questions that had failed: asked again
by them, most were answered. The web page shows the same list under "How to ask",
beside the question box.

1. **Name what you mean.** Say the unit, company or report: "NGIC's managing
   director", not "who runs the gas pipes". Broad questions fail most.
2. **Ask for a person by their role.** Use the title or the unit's initials: "Who is
   NGPIS's Chief Gas and Power Investment Officer?"
3. **One thing at a time works best.** Ask for a long list group by group, and
   compare two things in two questions.
4. **In a longer conversation, name the subject again.** After a few questions, or
   when you change topic, say "NNPC HMO's growth phase", not "its".
5. **Ask what the documents say, not what to do.** "Does NNPC HMO list occupational
   health?" gets an answer; "Should we sign their contract?" gets an advice
   referral.

When it cannot answer, the reply says so, shows what it searched for, lists the
closest pages and the scanned pages or pictures it could not quote, and offers
these tips.

## Serve

```sh
npm --prefix web run build                     # the web page, once (web/.output/public)
uv run --env-file .env petroknowledge serve   # the page and the JSON API on http://127.0.0.1:8091
```

The page is a conversation: each question shows its stages as they run, then the
verified answer with its dated sources, and a follow-up continues the conversation.
`npm --prefix web run dev` serves the page with live reloading on port 3000 and passes
its API requests to a running `serve`.

The API gives the same answer as `ask`, as JSON:
- `POST /api/v1/answers` with `{"question": "..."}` returns the answer, the id of its
  audit record and its place in a conversation.
- `POST /api/v1/answers/stream` sends the same answer as server-sent events: each
  stage as it starts, then the verified answer. Unverified text is never sent.
- A follow-up sends back `conversation_id`. The server keeps the conversation and
  rebuilds its history from what it showed, so a request carrying a history is refused.
- `GET /healthz` checks the process; `GET /readyz` checks the database and the three
  model servers.

To let invited people try it while this machine runs it, a Cloudflare quick tunnel
gives a temporary public link without an account
(`cloudflared tunnel --url http://127.0.0.1:8091`). It has no uptime guarantee, and
the link changes each time the tunnel starts.

## History

```sh
uv run --env-file .env petroknowledge history               # recent conversations and unanswered questions
uv run --env-file .env petroknowledge history CONVERSATION  # one conversation in full, as JSON
uv run --env-file .env petroknowledge history --export history.jsonl
```

Every answer is stored with each model request exactly as sent and the model's reply,
and a question that got no answer is stored with its cause. History is read only on
this machine: the API never serves it.

## Tests

```sh
uv run python scripts/check.py   # format, lint, types, tests at 100% coverage, attribution scan
```

The same command checks the web page: lint (with accessibility rules), a design
detector for generic model-default patterns, types, unit tests at 100% coverage, and
end-to-end tests in a real browser (WCAG 2.2 AA by axe in light and dark, keyboard
focus never hidden, the phone and high-zoom layouts).

The tests use invented hosts, pages, payloads and documents, and never reach the
network or run a model. They check the collection rules:
- robots.txt, including missing, broken and unreachable files;
- crawl-delay;
- size limits;
- off-host redirects;
- de-duplication by hash;
- resumable runs.

They also check the reading rules: host rules, Nuxt data, page validation and
retries in child processes.

The index tests run against the development Postgres (above), in a throwaway database
created and dropped by the test session, so it must be running.

## Licence

MIT; see `LICENSE`. Collected content keeps its owners' terms and is not covered by
this licence.
