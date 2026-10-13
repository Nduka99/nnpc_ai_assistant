import { expect, test } from "vitest";
import { createParser, type ServerEvent } from "./sse";

function parseAll(chunks: string[]): ServerEvent[] {
  const events: ServerEvent[] = [];
  const parse = createParser((event) => events.push(event));
  chunks.forEach(parse);
  return events;
}

const STREAM =
  'event: stage\ndata: {"stage": "searching"}\n\n' +
  ": keep-alive\n\n" +
  'event: answer\ndata: {"id": 7}\n\n';

test("events are the same however the stream is split into chunks", () => {
  const whole = parseAll([STREAM]);
  const byCharacter = parseAll([...STREAM]);

  expect(whole).toEqual([
    { event: "stage", data: '{"stage": "searching"}' },
    { event: "answer", data: '{"id": 7}' },
  ]);
  expect(byCharacter).toEqual(whole);
});

test("a CRLF line break split between two chunks is still one break", () => {
  expect(parseAll(["event: stage\r", "\ndata: 1\r\n\r", "\n"])).toEqual([
    { event: "stage", data: "1" },
  ]);
});

test("data lines join, unnamed events are messages, and empty events are dropped", () => {
  const events = parseAll(["data: one\ndata:two\nretry\n\nevent: empty\n\n"]);

  expect(events).toEqual([{ event: "message", data: "one\ntwo" }]);
});

test("an unfinished event waits for its blank line", () => {
  expect(parseAll(["event: stage\ndata: 1\n"])).toEqual([]);
});
