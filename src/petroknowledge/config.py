"""Every fixed setting in one place.

Paths are relative to the repository root, where every command is run. Everything
under data/ is collected from public sources and never committed or published: the
companies' terms allow a private research copy at most.
"""

import os
from pathlib import Path

# One folder per company (companies/<id>/: its plan and sources.json) and one file
# for the news feeds every company shares.
COMPANIES_DIR = Path("companies")
NEWS_SOURCES = COMPANIES_DIR / "news.json"
# How each host's pages are read (webpage): main content, furniture, headings.
HOST_RULES = COMPANIES_DIR / "hosts.json"
# Each stored web page read into elements (petroknowledge.pages); never committed.
PAGE_READINGS = Path("data/pages")
# Collected files (pages, documents, pictures), each stored once by SHA-256, with
# manifest.json recording where and when each came from.
FILE_STORE = Path("data/files")
# The link tier: metadata of often-updated pages, fetched live when answering.
LINK_REGISTRY = Path("data/links")
# Sizes measured before a download is approved.
MEASUREMENTS = Path("data/measurements")
# Files a person saved by hand, where a host's robots.txt forbids crawlers:
# data/inbox/<company>/, each named as at the end of its link.
INBOX = Path("data/inbox")

# Contact details are required by Wikimedia's User-Agent policy and are good manners
# for every host: the maintainer's public GitHub profile, no email.
USER_AGENT = (
    "petroknowledge/0.1 (https://github.com/Nduka99; independent research prototype)"
)
REQUEST_DELAY_SECONDS = 1.0  # at least this long between requests to one host
REQUEST_TIMEOUT_SECONDS = 30.0
REQUEST_RETRIES = 2

# What a page links that is worth reading: documents and pictures.
DOCUMENT_TYPES = (".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".csv")
# Each stored PDF read into elements (petroknowledge.documents); never committed.
READINGS = Path("data/elements")
# Scanned PDF pages read by a vision model (petroknowledge.transcribe); never committed.
TRANSCRIPTS = Path("data/transcripts")
# Pictures stored once as PNG with the text read in them (petroknowledge.images).
IMAGES = Path("data/images")
# SigLIP2 so400m ranks pictures by what they show (petroknowledge.siglip).
SIGLIP = Path("models/siglip2-so400m-patch14-384")
SIGLIP_TEXT_TOKENS = 64  # a question padded as the model card does

# The local model servers (llama.cpp), settings carried over from the Lime Green
# assistant. Qwen3-Embedding-4B runs on the CPU, its vectors cut to the index's size.
EMBEDDING_URL = os.environ.get(
    "PETROKNOWLEDGE_EMBEDDING_URL", "http://127.0.0.1:8084/v1/embeddings"
)
EMBEDDING_MODEL = os.environ.get(
    "PETROKNOWLEDGE_EMBEDDING_MODEL", "Qwen3-Embedding-4B-Q8_0.gguf (first 1024)"
)
# The index column's size. A model trained for shorter vectors (Qwen3-Embedding's
# Matryoshka training) gives longer ones that are cut to this size and normalised.
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 16  # texts per embedding request (Lime Green's setting)
# Qwen3-Embedding expects an instruction before a query and nothing before a passage.
# Lime Green's wording, made general so every company's index shares it.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a Nigerian oil and gas company, "
    "retrieve the public passages that answer it\nQuery: "
)
RERANK_URL = "http://127.0.0.1:8082/v1/rerank"
# The model servers' key (llama-server --api-key), a secret, so it comes from the
# environment (.env). Without it no key is sent.
MODEL_API_KEY = os.environ.get("PETROKNOWLEDGE_MODEL_API_KEY", "")
SEARCH_TIMEOUT_SECONDS = 120.0  # embedding or reranking one request
CHAT_URL = os.environ.get(
    "PETROKNOWLEDGE_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions"
)
# The generator keeps its expert layers in system RAM, so an answer can take a while.
CHAT_TIMEOUT_SECONDS = 300.0
# The grounding check's judge: a model of another family than the generator, so
# the claims are not judged by the model that wrote them.
JUDGE_URL = os.environ.get(
    "PETROKNOWLEDGE_JUDGE_URL", "http://127.0.0.1:8085/v1/chat/completions"
)
# Deterministic sampling, so the same question gets a repeatable answer.
TEMPERATURE = 0.0
SEED = 42
# A reused prompt cache changes the batch sizes a prompt is computed in, and the
# logits are then not bit-for-bit identical (llama.cpp server README), so the same
# request could get a different answer. Every request is computed whole.
CACHE_PROMPT = False
MAX_ANSWER_TOKENS = 2048
# A model server's /health answers at once when its model is loaded.
HEALTH_TIMEOUT_SECONDS = 2.0

# The index (Lime Green's measured settings). A passage holds at most this
# many characters: a PDF page's sections or a web page's sections packed together.
MAX_PASSAGE_CHARS = 1500
# Keyword and vector candidates before fusion, and fused candidates the reranker
# orders: with 20, the current units list, at vector rank 13-55 and with no
# keyword hit, never reached the reranker, which ranks it first when it does.
# Practitioners rerank 50-100.
CANDIDATES_PER_METHOD = 50
RRF_K = 60  # the standard reciprocal-rank-fusion constant
RERANK_CANDIDATES = 50
TOP_K = 8
# The answer path (Lime Green's measured settings). The model returns at most
# MAX_CLAIMS claims of at most MAX_QUOTES_PER_CLAIM quotes each.
MAX_CLAIMS = 8
MAX_QUOTES_PER_CLAIM = 3
# A question is searched as at most MAX_PARTS separate things; a question with several
# gets up to MAX_PASSAGES passages, interleaved from each search (Lime Green S2b C2).
MAX_PARTS = 6
MAX_PASSAGES = 12
# Passages given to the answer model, at most (Lime Green X39: evidence use falls
# beyond it).
PASSAGE_BUDGET = 32
# A part may list several things (at most MAX_ITEMS), each searched alone, adding its
# best ITEM_TOP passages not yet given (Lime Green X48).
MAX_ITEMS = 6
ITEM_TOP = 4
# After each search, a search inside the units the user's words name adds its best
# SCOPED_TOP passages, within the budget (Lime Green E5 B4).
SCOPED_TOP = 4
MAX_PICTURES = 4  # picture passages added to an answer, at most (Lime Green X43 B4)
CLOSEST_PAGES = 3  # pages listed with the insufficient-evidence text
# Search channels (Lime Green X44 F2): pictures and guidance are ranked
# apart and added after the company's places, never in them. Each search adds its
# best PICTURES_PER_SEARCH pictures by their words and by what they show, and up to
# GUIDANCE_PER_SEARCH guidance passages the reranker puts above the company's last.
PICTURES_PER_SEARCH = 2
GUIDANCE_PER_SEARCH = 2
# Conversations (Lime Green ADR 0035): the first request reads the last HISTORY_TURNS
# turns as the user saw them (X36 measured 4). A conversation ends after
# CONVERSATION_IDLE_MINUTES without a turn, CONVERSATION_MAX_HOURS after it started
# (OWASP session management) or at CONVERSATION_MAX_TURNS turns (OWASP LLM10).
HISTORY_TURNS = 4
CONVERSATION_IDLE_MINUTES = 30
CONVERSATION_MAX_HOURS = 8
CONVERSATION_MAX_TURNS = 20
# Postgres with pgvector and pg_textsearch (deploy/compose.yaml); a secret, so it
# comes from the environment (.env). Empty: no database is reached.
DATABASE_URL = os.environ.get("PETROKNOWLEDGE_DATABASE_URL", "")
# The database is on this machine; psycopg's own default waits 130 s before failing.
DATABASE_CONNECT_TIMEOUT_SECONDS = 5
# An index version to serve in place of the live one, so a candidate can be evaluated
# before it goes live. Unset, the live version is served.
INDEX_VERSION = os.environ.get("PETROKNOWLEDGE_INDEX_VERSION", "")
# The grounding check removes the claims its judge finds unsupported: "figures"
# judges only claims stating a figure (chosen by measurement: it kept the judge's
# catches of substituted figures and gave back a third of the refusals), "remove" every
# claim, and "record" keeps every claim, for an evaluation run: the verdicts are
# recorded either way, so one run gives every arm, the removals applied offline.
GROUNDING = os.environ.get("PETROKNOWLEDGE_GROUNDING", "figures")

# The API (`petroknowledge serve`), on this machine only. Lime Green's app uses 8090,
# so this port keeps the two projects independent (D07).
APP_HOST = "127.0.0.1"
APP_PORT = 8091
# The web app's build (`npm run build` in web/), served by the API.
WEB_DIST = Path("web/.output/public")
# The API refuses longer questions (OWASP LLM10, unbounded consumption). The longest
# question in the development and scope sets is 152 characters.
MAX_QUESTION_CHARS = 1000
MAX_FEEDBACK_CHARS = 1000  # a reader's comment on an answer
# At most as many answers at once as the generator has slots; a few more wait in
# line, and one client may ask a few questions a minute.
ANSWERS_AT_ONCE = 2
QUEUE_LENGTH = 6
QUESTIONS_PER_MINUTE = 10
