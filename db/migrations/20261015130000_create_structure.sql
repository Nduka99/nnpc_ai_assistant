-- The company's structure as its own units page lists it (K30 G1): each group's
-- members in the page's order, linked to the unit where the member is one. With the
-- units, their former names and the dated role statements, it is the graph the
-- answer path is told and checks claims against.

-- migrate:up
CREATE TABLE structure (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id bigint NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    position integer NOT NULL,       -- the member's place in the page's list
    group_name text NOT NULL,
    member text NOT NULL,            -- as the page lists it
    unit_id bigint REFERENCES units (id) ON DELETE CASCADE,  -- when it is a unit
    UNIQUE (index_version_id, position)
);

-- migrate:down
DROP TABLE structure;
