-- Provenance and history (ADR 0016). Every request sent to the generator while
-- answering (the understanding request, any second rewrite, the answer request) is
-- kept with the model's reply, so a reviewer can see exactly what produced an
-- answer; it is written in the same transaction as the answer. A question that got
-- no answer is kept too, with its cause. Both are read only on this machine
-- (`petroknowledge history`), never served.

-- migrate:up
CREATE TABLE model_calls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    answer_id bigint NOT NULL REFERENCES answers (id) ON DELETE CASCADE,
    position integer NOT NULL,       -- 1 for the answer's first request
    purpose text NOT NULL CHECK (purpose IN ('understanding', 'answering')),
    system text NOT NULL,            -- the fixed system prompt as sent
    user_message text NOT NULL,      -- the request's user message: question, history, passages
    reply jsonb NOT NULL,            -- the model's structured reply, before verification
    seconds real NOT NULL,
    UNIQUE (answer_id, position)
);

-- The conversation is as the client sent it: it may be unknown or ended, so it has
-- no foreign key.
CREATE TABLE question_failures (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    question text NOT NULL,
    conversation_id uuid,
    turn integer,
    reason text NOT NULL CHECK (reason IN ('ended', 'unavailable')),
    detail text NOT NULL             -- the server-side cause, never sent to the client
);

CREATE INDEX question_failures_created_at ON question_failures (created_at);

-- migrate:down
DROP TABLE question_failures;
DROP TABLE model_calls;
