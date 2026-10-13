import { settled, type TurnState } from "./thread";

// The conversation is kept in this tab's session storage, so a reload, a Back from
// a source or a discarded phone tab does not lose it (Zeigarnik: interrupted work
// stays resumable). The server still decides whether the conversation continues.

const KEY = "petroknowledge.thread";

export interface SavedThread {
  turns: TurnState[];
  conversationId: string | null;
}

/** The saved thread, with any answer that was still on its way marked stopped, or
 * an empty thread when nothing usable is saved or storage is unavailable. */
export function loadThread(): SavedThread {
  try {
    const saved = JSON.parse(
      sessionStorage.getItem(KEY) ?? "null",
    ) as SavedThread | null;
    if (saved === null || !Array.isArray(saved.turns)) {
      return { turns: [], conversationId: null };
    }
    return {
      turns: settled(saved.turns),
      conversationId: saved.conversationId ?? null,
    };
  } catch {
    return { turns: [], conversationId: null };
  }
}

/** Keep the thread; a full or blocked storage only loses the convenience. */
export function saveThread(thread: SavedThread): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(thread));
  } catch {
    // Nothing to do: the thread still works in memory.
  }
}
