-- Business units (K23): the units a company's own units page lists and links, and
-- the passages in each unit's scope, kept per index version. The graph's first
-- entities (K15).

-- migrate:up
CREATE TABLE units (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id bigint NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    name text NOT NULL,              -- as the units page lists it
    acronym text NOT NULL DEFAULT '',
    url text NOT NULL                -- the unit's own page
);

CREATE INDEX units_index_version ON units (index_version_id);

-- A passage is in a unit's scope when its document is the unit's page, or when its
-- title or its text names the unit (`units.members`).
CREATE TABLE unit_passages (
    unit_id bigint NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    passage_id bigint NOT NULL REFERENCES passages (id) ON DELETE CASCADE,
    PRIMARY KEY (unit_id, passage_id)
);

-- migrate:down
DROP TABLE unit_passages;
DROP TABLE units;
