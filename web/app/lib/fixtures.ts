// Invented answers for the tests, in the API's shapes.
import type { AnswerView, RecordedAnswer } from "./api";

export function view(change: Partial<AnswerView> = {}): AnswerView {
  return {
    question: "Where does NETCO's school outreach take place?",
    status: "answered",
    notice: "",
    claims: [{ text: "NETCO visits secondary schools.", sources: [1] }],
    sources: [
      {
        number: 1,
        title: "NNPC Engineering & Technical Company Limited (NETCO)",
        heading: "Community",
        quote: "NETCO visits secondary schools.",
        link: "https://nnpcgroup.com/netco#:~:text=NETCO%20visits",
        url: "https://nnpcgroup.com/netco",
        published: "",
        captured: "2026-10-10",
        compiled: false,
      },
    ],
    closest_pages: [],
    pointers: [],
    ...change,
  };
}

export function recorded(
  change: Partial<RecordedAnswer> = {},
  answer: Partial<AnswerView> = {},
): RecordedAnswer {
  return {
    id: 7,
    conversation_id: "c-1",
    turn: 1,
    understood_as: ["Where does NETCO's school outreach take place?"],
    answer: view(answer),
    ...change,
  };
}
