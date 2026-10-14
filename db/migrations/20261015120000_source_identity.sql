-- What a source is and when it was written. A document's file date is read
-- from its file's own metadata (a PDF's creation date): it dates a document whose
-- publication date is unknown, and is shown to the answer request as "file dated",
-- never as a publication date. A unit's former names are those its own page gives
-- it ("formerly the Nigerian Petroleum Development Company (NPDC)"), so a question
-- using one names the unit.

-- migrate:up
ALTER TABLE documents ADD COLUMN file_date date;
ALTER TABLE units ADD COLUMN aliases text NOT NULL DEFAULT '';  -- "; " between names

-- migrate:down
ALTER TABLE units DROP COLUMN aliases;
ALTER TABLE documents DROP COLUMN file_date;
