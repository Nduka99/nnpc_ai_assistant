-- The grounding check's requests (K28) are model calls too: one per verified claim,
-- kept with the answer like the others.

-- migrate:up
ALTER TABLE model_calls DROP CONSTRAINT model_calls_purpose_check;
ALTER TABLE model_calls ADD CONSTRAINT model_calls_purpose_check
    CHECK (purpose IN ('understanding', 'answering', 'grounding'));

-- migrate:down
ALTER TABLE model_calls DROP CONSTRAINT model_calls_purpose_check;
ALTER TABLE model_calls ADD CONSTRAINT model_calls_purpose_check
    CHECK (purpose IN ('understanding', 'answering'));
