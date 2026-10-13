import type { components } from "./api-types";
import { createParser } from "./sse";

export type RecordedAnswer = components["schemas"]["RecordedAnswer"];
export type AnswerView = components["schemas"]["AnswerView"];
export type Source = components["schemas"]["Source"];

export const STAGES = ["understanding", "searching", "answering", "checking"] as const;
export type Stage = (typeof STAGES)[number];

/** What happens to one question: its place in line if it must wait, each stage as
 * it starts, then one outcome. */
export type StreamEvent =
  | { kind: "queued"; position: number }
  | { kind: "stage"; stage: Stage }
  | { kind: "answer"; recorded: RecordedAnswer }
  | { kind: "error"; detail: string }
  | { kind: "ended"; detail: string }
  | { kind: "stopped" };

// The server's fixed texts (`app.UNAVAILABLE`, `app.ENDED`), used when a reply
// carries none, such as a dropped connection.
export const UNAVAILABLE =
  "The assistant can't answer right now. Try again in a moment.";
export const ENDED = "This conversation has ended. Please start a new conversation.";
export const BUSY = "The assistant is busy. Try again in a minute.";

/**
 * Ask one question over the v1 stream (`POST /api/v1/answers/stream`), continuing
 * `conversationId` if given. The server keeps the conversation and rebuilds its
 * history, so the request carries only the question and the id. Aborting `signal`
 * stops the request, and the question ends as "stopped" rather than as a failure.
 */
export async function askStream(
  question: string,
  conversationId: string | null,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const fail = () =>
    onEvent(
      signal?.aborted ? { kind: "stopped" } : { kind: "error", detail: UNAVAILABLE },
    );
  let response: Response;
  try {
    response = await fetchImpl("/api/v1/answers/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(
        conversationId === null
          ? { question }
          : { question, conversation_id: conversationId },
      ),
      signal,
    });
  } catch {
    fail();
    return;
  }
  if (response.status === 404) {
    onEvent({ kind: "ended", detail: ENDED });
    return;
  }
  if (response.status === 429) {
    onEvent({ kind: "error", detail: BUSY });
    return;
  }
  if (!response.ok || response.body === null) {
    onEvent({ kind: "error", detail: UNAVAILABLE });
    return;
  }
  let finished = false;
  const parse = createParser(({ event, data }) => {
    const payload: unknown = JSON.parse(data);
    if (event === "queued") {
      onEvent({ kind: "queued", position: (payload as { position: number }).position });
    } else if (event === "stage") {
      onEvent({ kind: "stage", stage: (payload as { stage: Stage }).stage });
    } else if (event === "answer") {
      finished = true;
      onEvent({ kind: "answer", recorded: payload as RecordedAnswer });
    } else if (event === "error") {
      finished = true;
      onEvent({ kind: "error", detail: UNAVAILABLE });
    }
  });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parse(value);
    }
  } catch {
    // A dropped connection or a stop: the outcome below says which.
  }
  if (!finished) fail();
}

/**
 * Send a reader's judgement of one answer (`POST /api/v1/answers/{id}/feedback`),
 * from the conversation the answer belongs to. Whether it was kept.
 */
export async function sendFeedback(
  answerId: number,
  conversationId: string,
  helpful: boolean,
  comment: string,
  fetchImpl: typeof fetch = fetch,
): Promise<boolean> {
  try {
    const response = await fetchImpl(`/api/v1/answers/${answerId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId, helpful, comment }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
