import type { RecordedAnswer, Stage } from "./api";

/** How one question ended: a verified answer, the fixed error, an ended
 * conversation, or stopped by the reader. Absent while the answer is on its way. */
export type Outcome =
  | { kind: "answer"; recorded: RecordedAnswer }
  | { kind: "error"; detail: string }
  | { kind: "ended"; detail: string }
  | { kind: "stopped" };

export interface TurnState {
  key: number;
  question: string;
  stages: Stage[]; // the stages started so far, in order
  queued?: number; // its place in line while it waits for the assistant
  outcome?: Outcome;
}

/** Turns brought back from storage or an undo: an answer that was still on its
 * way can no longer arrive, so it is marked as not finished. */
export function settled(turns: TurnState[]): TurnState[] {
  return turns.map((turn) =>
    turn.outcome ? turn : { ...turn, outcome: { kind: "stopped" as const } },
  );
}

/** What a screen reader hears about the latest turn (WCAG 4.1.3): the outcome once
 * it arrives. Errors are announced by their own alert. */
export function announcement(turn: TurnState | undefined): string {
  if (turn === undefined) return "";
  const outcome = turn.outcome;
  if (outcome === undefined) return "Working on your question.";
  if (outcome.kind === "stopped") return "Answer stopped.";
  if (outcome.kind !== "answer") return "";
  const answer = outcome.recorded.answer;
  if (answer.status === "safety_referral") return "Advice referral.";
  if (answer.status !== "answered") return "Not enough information was found.";
  const claims = answer.claims.length;
  const sources = answer.sources.length;
  return `Answer ready: ${claims} ${claims === 1 ? "statement" : "statements"} from ${sources} ${sources === 1 ? "source" : "sources"}.`;
}
