-- The index and the answer log: the Lime Green assistant's schema as its migrations
-- left it, in one step, with what PetroKnowledge's sources add (a passage's kind,
-- whether it may be quoted, a post's publication date).

-- migrate:up
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- BM25 keyword search keeps stop words (Lime Green X2): names such as "The Petroleum
-- Industry Act" or "Who we are" are searched as written.
CREATE TEXT SEARCH DICTIONARY english_stem_keep_stop (TEMPLATE = snowball, Language = english);
CREATE TEXT SEARCH CONFIGURATION english_keep_stop (COPY = english);
ALTER TEXT SEARCH CONFIGURATION english_keep_stop
    ALTER MAPPING REPLACE english_stem WITH english_stem_keep_stop;

CREATE TABLE documents (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url text NOT NULL,
    title text NOT NULL,
    fetched_at timestamptz NOT NULL,
    sha256 text NOT NULL,  -- of the exact collected bytes
    UNIQUE (url, sha256)   -- the same capture is stored once; a changed one is a new row
);

CREATE TABLE index_versions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    built_at timestamptz NOT NULL DEFAULT now(),
    corpus_sha256 text NOT NULL,    -- sorted (url, file sha256) pairs
    passages_sha256 text NOT NULL,  -- every passage as indexed: compares builds
    embedding_model text NOT NULL,  -- the API refuses to serve with a different model
    live boolean NOT NULL DEFAULT false
);

CREATE UNIQUE INDEX index_versions_one_live ON index_versions (live) WHERE live;

CREATE TABLE images (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),  -- the PNG's SHA-256
    png bytea NOT NULL,
    siglip vector(1152)  -- SigLIP2's picture vector (Lime Green X44 F2)
);

CREATE TABLE passages (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id bigint NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    document_id bigint NOT NULL REFERENCES documents (id),
    title text NOT NULL,          -- the page's or document's title, searched with the text
    heading text NOT NULL,
    text text NOT NULL,           -- the source's own words: the only part quoted
    context text NOT NULL DEFAULT '',  -- searched and embedded, never quoted
    page integer CHECK (page > 0),     -- a document passage's page
    kind text NOT NULL CHECK (kind IN ('page', 'post', 'pdf', 'sheet', 'scan', 'picture')),
    channel text NOT NULL CHECK (channel IN ('company', 'guidance', 'picture')),
    -- A scanned page's transcript is searched but never quoted: no transcription
    -- passed its accuracy gate.
    quotable boolean NOT NULL DEFAULT true,
    published date,               -- a post's publication date
    image text REFERENCES images (id),
    -- Unit length, so the inner product is the cosine similarity. Exact search: a few
    -- thousand passages are fast to scan, with perfect recall.
    embedding vector(1024) NOT NULL
);

CREATE INDEX passages_index_version ON passages (index_version_id);

CREATE TABLE conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    last_active_at timestamptz NOT NULL DEFAULT now(),
    turns integer NOT NULL DEFAULT 0  -- turns reserved so far, failed ones included
);

CREATE INDEX conversations_created_at ON conversations (created_at);

CREATE TABLE answers (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    question text NOT NULL,
    status text NOT NULL
        CHECK (status IN ('answered', 'insufficient_evidence', 'safety_referral')),
    shown jsonb NOT NULL,           -- the reader's view
    removed jsonb NOT NULL,         -- claims removed by verification: text and reason
    passage_ids bigint[] NOT NULL,  -- the passages the model was given, best first
    index_version_id bigint NOT NULL,
    embedding_model text NOT NULL,
    prompt_sha256 text NOT NULL,    -- the fixed system prompts in use
    seconds real NOT NULL,          -- time to answer
    conversation_id uuid REFERENCES conversations (id),
    turn integer,
    search_questions jsonb NOT NULL DEFAULT '[]',
    CONSTRAINT answers_turn_with_conversation
        CHECK ((conversation_id IS NULL) = (turn IS NULL))
);

CREATE INDEX answers_created_at ON answers (created_at);
CREATE UNIQUE INDEX answers_conversation_turn ON answers (conversation_id, turn);

-- migrate:down
DROP TABLE answers;
DROP TABLE conversations;
DROP TABLE passages;
DROP TABLE images;
DROP TABLE index_versions;
DROP TABLE documents;
DROP TEXT SEARCH CONFIGURATION english_keep_stop;
DROP TEXT SEARCH DICTIONARY english_stem_keep_stop;
DROP EXTENSION pg_textsearch;
DROP EXTENSION vector;
