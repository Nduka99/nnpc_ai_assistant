import { afterEach, expect, test, vi } from "vitest";
import { loadThread, saveThread } from "./saved";

afterEach(() => {
  sessionStorage.clear();
  vi.restoreAllMocks();
});

test("a saved thread comes back, with an unfinished answer marked stopped", () => {
  saveThread({
    conversationId: "c-1",
    turns: [
      {
        key: 1,
        question: "Who leads NETCO?",
        stages: [],
        outcome: { kind: "stopped" },
      },
      { key: 2, question: "Since when?", stages: ["searching"] },
    ],
  });

  const loaded = loadThread();

  expect(loaded.conversationId).toBe("c-1");
  expect(loaded.turns.map((turn) => turn.outcome)).toEqual([
    { kind: "stopped" },
    { kind: "stopped" },
  ]);
});

test("nothing saved, unusable data or blocked storage give an empty thread", () => {
  const empty = { turns: [], conversationId: null };
  expect(loadThread()).toEqual(empty);

  sessionStorage.setItem("petroknowledge.thread", '{"turns": "no"}');
  expect(loadThread()).toEqual(empty);

  sessionStorage.setItem("petroknowledge.thread", '{"turns": []}');
  expect(loadThread()).toEqual(empty);

  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  expect(loadThread()).toEqual(empty);
});

test("a storage that refuses to save is ignored", () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("full");
  });

  expect(() => saveThread({ turns: [], conversationId: null })).not.toThrow();
});
