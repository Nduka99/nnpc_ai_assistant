import { expect, test, vi } from "vitest";
import {
  askStream,
  BUSY,
  ENDED,
  sendFeedback,
  UNAVAILABLE,
  type StreamEvent,
} from "./api";
import { recorded } from "./fixtures";

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
}

function respond(body: BodyInit | null, status = 200): typeof fetch {
  return vi.fn(async () => new Response(body, { status }));
}

async function events(fetchImpl: typeof fetch, id: string | null = null) {
  const heard: StreamEvent[] = [];
  await askStream(
    "Who leads NETCO?",
    id,
    (event) => heard.push(event),
    undefined,
    fetchImpl,
  );
  return heard;
}

const ANSWER = recorded();
const STREAM =
  'event: stage\ndata: {"stage":"understanding"}\n\n' +
  `event: answer\ndata: ${JSON.stringify(ANSWER)}\n\n`;

test("a first question sends only the question and hears each stage, then the answer", async () => {
  const fetchImpl = respond(streamOf(STREAM));

  expect(await events(fetchImpl)).toEqual([
    { kind: "stage", stage: "understanding" },
    { kind: "answer", recorded: ANSWER },
  ]);
  const [url, init] = vi.mocked(fetchImpl).mock.calls[0] as [string, RequestInit];
  expect(url).toBe("/api/v1/answers/stream");
  expect(JSON.parse(init.body as string)).toEqual({ question: "Who leads NETCO?" });
});

test("a follow-up sends the conversation id and nothing else", async () => {
  const fetchImpl = respond(streamOf(STREAM));

  await events(fetchImpl, "c-1");

  const [, init] = vi.mocked(fetchImpl).mock.calls[0] as [string, RequestInit];
  expect(JSON.parse(init.body as string)).toEqual({
    question: "Who leads NETCO?",
    conversation_id: "c-1",
  });
});

test("an ended conversation is its own outcome", async () => {
  expect(await events(respond('{"detail": "ended"}', 404))).toEqual([
    { kind: "ended", detail: ENDED },
  ]);
});

test("every failure shows the fixed message, never the server's detail", async () => {
  const unreachable: typeof fetch = vi.fn(async () => {
    throw new TypeError("Failed to fetch");
  });
  const broken = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.error(new Error("connection reset"));
    },
  });
  const failures: (typeof fetch)[] = [
    respond('{"detail": "internal address"}', 503),
    respond(null),
    unreachable,
    respond(streamOf('event: error\ndata: {"detail": "x"}\n\n')),
    // An event the client does not know is ignored; the stream then ends unanswered.
    respond(
      streamOf(
        'event: stage\ndata: {"stage":"searching"}\n\n',
        "event: ping\ndata: {}\n\n",
      ),
    ),
    respond(broken),
  ];

  for (const fetchImpl of failures) {
    expect((await events(fetchImpl)).at(-1)).toEqual({
      kind: "error",
      detail: UNAVAILABLE,
    });
  }
});

test("a stopped question ends as stopped, whether it was waiting or streaming", async () => {
  const waiting = new AbortController();
  const unanswered: typeof fetch = vi.fn(
    (_url: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("stopped", "AbortError")),
        );
      }),
  );
  const streaming = new AbortController();
  const stalled = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode('event: stage\ndata: {"stage":"searching"}\n\n'),
      );
      streaming.signal.addEventListener("abort", () =>
        controller.error(new DOMException("stopped", "AbortError")),
      );
    },
  });

  for (const [controller, fetchImpl] of [
    [waiting, unanswered],
    [streaming, respond(stalled)],
  ] as const) {
    const heard: StreamEvent[] = [];
    const asked = askStream(
      "Who leads NETCO?",
      null,
      (e) => heard.push(e),
      controller.signal,
      fetchImpl,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();
    await asked;
    expect(heard.at(-1)).toEqual({ kind: "stopped" });
  }
});

test("a question that must wait hears its place in line, and a full line is busy", async () => {
  const queued = respond(streamOf('event: queued\ndata: {"position": 2}\n\n' + STREAM));

  expect((await events(queued))[0]).toEqual({ kind: "queued", position: 2 });
  expect(await events(respond('{"detail": "busy"}', 429))).toEqual([
    { kind: "error", detail: BUSY },
  ]);
});

test("feedback is sent with its conversation, and a failure is reported, not thrown", async () => {
  const kept = respond(null, 204);

  expect(await sendFeedback(7, "c-1", false, "wrong year", kept)).toBe(true);
  const [url, init] = vi.mocked(kept).mock.calls[0] as [string, RequestInit];
  expect(url).toBe("/api/v1/answers/7/feedback");
  expect(JSON.parse(init.body as string)).toEqual({
    conversation_id: "c-1",
    helpful: false,
    comment: "wrong year",
  });
  expect(await sendFeedback(7, "c-2", true, "", respond('{"detail": "x"}', 404))).toBe(
    false,
  );
  const unreachable: typeof fetch = vi.fn(async () => {
    throw new TypeError("Failed to fetch");
  });
  expect(await sendFeedback(7, "c-1", true, "", unreachable)).toBe(false);
});
