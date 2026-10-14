-- Feedback on answers: a reader marks an answer helpful or not right, with an
-- optional comment, so testers' judgements become evaluation material. It is only
-- accepted from the conversation the answer belongs to, and read only on this
-- machine (`petroknowledge history`), never served.

-- migrate:up
CREATE TABLE answer_feedback (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    answer_id bigint NOT NULL REFERENCES answers (id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    helpful boolean NOT NULL,
    comment text NOT NULL DEFAULT '' CHECK (length(comment) <= 1000)
);

CREATE INDEX answer_feedback_answer ON answer_feedback (answer_id);

-- migrate:down
DROP TABLE answer_feedback;
