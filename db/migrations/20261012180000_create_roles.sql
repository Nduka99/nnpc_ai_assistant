-- Dated roles (K24): who holds which role, as each dated passage states it, so code
-- can decide which statement is the latest. A passage's reading is kept by its text,
-- across versions, so a later build or refresh reads only new passages.

-- migrate:up
CREATE TABLE role_readings (
    passage_sha256 text PRIMARY KEY CHECK (passage_sha256 ~ '^[0-9a-f]{64}$'),
    statements jsonb NOT NULL,       -- (person, role, organisation, quote), kept ones
    read_at timestamptz NOT NULL DEFAULT now()
);

-- A statement of an index version: its passage, and the date it is as of (a post's
-- publication date, else a page's capture date).
CREATE TABLE role_statements (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id bigint NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    passage_id bigint NOT NULL REFERENCES passages (id) ON DELETE CASCADE,
    person text NOT NULL,
    role text NOT NULL,
    organisation text NOT NULL,
    quote text NOT NULL,             -- the passage's own words
    stated_on date NOT NULL,
    date_kind text NOT NULL CHECK (date_kind IN ('published', 'captured'))
);

CREATE INDEX role_statements_index_version ON role_statements (index_version_id);

-- migrate:down
DROP TABLE role_statements;
DROP TABLE role_readings;
