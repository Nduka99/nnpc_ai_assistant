import { render, screen, within } from "@testing-library/vue";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import type { StreamEvent } from "../lib/api";
import { recorded, view } from "../lib/fixtures";
import ThreadApp from "./ThreadApp.vue";

const MAX_QUESTION_CHARS = 1000;
/** A link's accessible name: its title, then the hidden new-tab warning. */
function newTab(title: string): RegExp {
  const literal = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${literal}\\s*\\(opens in a new tab\\)$`);
}
const NETCO = newTab("NNPC Engineering & Technical Company Limited (NETCO)");

interface Asked {
  question: string;
  conversationId: string | null;
  signal: AbortSignal | undefined;
  send: (event: StreamEvent) => void;
  finish: () => void;
}

/** A stand-in for the stream: each question waits until the test sends its
 * events and finishes it. */
function fakeAsk() {
  const asked: Asked[] = [];
  const ask = (
    question: string,
    conversationId: string | null,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ) =>
    new Promise<void>((finish) => {
      asked.push({ question, conversationId, signal, send: onEvent, finish });
    });
  return { ask, asked };
}

function start() {
  const user = userEvent.setup();
  const { ask, asked } = fakeAsk();
  const shown = render(ThreadApp, { props: { ask } });
  return { user, asked, shown };
}

async function askAndAnswer(
  user: ReturnType<typeof userEvent.setup>,
  asked: Asked[],
  question: string,
  outcome: StreamEvent,
) {
  await user.type(screen.getByLabelText("Your question"), `${question}{Enter}`);
  const latest = asked.at(-1) as Asked;
  latest.send(outcome);
  latest.finish();
  await screen.findByRole("button", { name: "Ask" });
}

test("the empty thread teaches the input and its examples ask themselves", async () => {
  const { user, asked } = start();

  expect(
    screen.getByRole("heading", { name: "Ask a question to start" }),
  ).toBeVisible();
  expect(screen.queryByRole("button", { name: "New conversation" })).toBeNull();
  expect(screen.getByText(/Questions and answers are recorded/)).toBeVisible();
  await user.click(
    screen.getByRole("button", {
      name: "Who is NNPC's Group Chief Executive Officer?",
    }),
  );

  expect(asked[0]).toMatchObject({
    question: "Who is NNPC's Group Chief Executive Officer?",
    conversationId: null,
  });
  // Focus waits in the question box, where Esc and the next question are.
  expect(screen.getByLabelText("Your question")).toHaveFocus();
  expect(
    await screen.findByRole("heading", {
      name: "Who is NNPC's Group Chief Executive Officer?",
    }),
  ).toBeVisible();
});

test("each stage is shown as it starts, then the claims and their quotes", async () => {
  const { user, asked } = start();

  await user.type(screen.getByLabelText("Your question"), "Where?{Enter}");
  expect(await screen.findByText(/Sending the question/)).toBeVisible();
  expect(screen.getByText("Answers usually take 15–30 seconds.")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("Working on your question.");
  asked[0]?.send({ kind: "stage", stage: "understanding" });
  asked[0]?.send({ kind: "stage", stage: "searching" });
  expect(await screen.findByText(/Searching NNPC's public pages/)).toBeVisible();
  expect(screen.getByText("(done)")).toBeInTheDocument();
  expect(screen.getByText("(in progress)")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Stop" })).toBeVisible();

  asked[0]?.send({ kind: "answer", recorded: recorded() });
  asked[0]?.finish();

  const ref = await screen.findByRole("link", { name: "Source 1" });
  expect(ref).toHaveAttribute("href", "#t1-s1");
  const sources = screen.getByRole("region", { name: "Sources" });
  const title = within(sources).getByRole("link", { name: NETCO });
  expect(title).toHaveAttribute(
    "href",
    "https://nnpcgroup.com/netco#:~:text=NETCO%20visits",
  );
  expect(title).toHaveAttribute("target", "_blank");
  expect(within(sources).getByText("Found word for word")).toBeVisible();
  expect(within(sources).getByText("Retrieved 10 Oct 2026")).toBeVisible();
  expect(within(sources).queryByText(/Published/)).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent(
    "Answer ready: 1 statement from 1 source.",
  );
  // "Understood as" is for follow-ups only.
  expect(screen.queryByText("Understood as")).toBeNull();
});

test("a follow-up continues the conversation and shows what it was understood as", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  const second = recorded(
    { turn: 2, understood_as: ["Who does NETCO's outreach reach?"] },
    {
      claims: [
        { text: "Secondary school students.", sources: [1] },
        { text: "In Lagos and Abuja.", sources: [1] },
      ],
    },
  );
  await askAndAnswer(user, asked, "Who?", { kind: "answer", recorded: second });

  expect(asked[1]?.conversationId).toBe("c-1");
  expect(screen.getByText("Who does NETCO's outreach reach?")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent(
    "Answer ready: 2 statements from 1 source.",
  );
  // Each turn's references point to its own sources.
  const refs = screen.getAllByRole("link", { name: "Source 1" });
  expect(refs.map((ref) => ref.getAttribute("href"))).toEqual([
    "#t1-s1",
    "#t2-s1",
    "#t2-s1",
  ]);
});

test("a dated PDF source shows its page, and unquotable pages are listed once each", async () => {
  const { user, asked } = start();
  const report = "https://cms1977.nnpcgroup.com/uploads/report.pdf";
  const dated = recorded(
    {},
    {
      sources: [
        {
          number: 1,
          title: "NNPC 2025 Annual Report",
          heading: "",
          quote: "Profit After Tax rose 33% to ₦7.2 trillion.",
          link: `${report}#page=7`,
          url: report,
          published: "2026-09-29",
          captured: "2026-10-10",
          compiled: false,
        },
        {
          number: 2,
          title: "Press release",
          heading: "Results",
          quote: "“Revenue stood at ₦34.5 trillion,” the release said.",
          link: "https://nnpcgroup.com/posts/profit",
          url: "https://nnpcgroup.com/posts/profit",
          published: "2026-09-29",
          captured: "2026-10-10",
          compiled: false,
        },
      ],
      pointers: [
        { title: "Annual Report", link: `${report}#page=12`, kind: "scanned page" },
        { title: "Annual Report", link: `${report}#page=12&x`, kind: "scanned page" },
        { title: "Annual Report", link: `${report}#page=5`, kind: "scanned page" },
        { title: "A chart", link: "https://example.com/chart.png", kind: "picture" },
      ],
    },
  );

  await askAndAnswer(user, asked, "Profit?", { kind: "answer", recorded: dated });

  const sources = screen.getByRole("region", { name: "Sources" });
  expect(within(sources).getAllByText("Published 29 Sep 2026")).toHaveLength(2);
  expect(within(sources).getByText("p. 7")).toBeVisible();
  // A quote that opens with its own mark gets no added one.
  const quotes = [...sources.querySelectorAll("blockquote")];
  expect(quotes.map((quote) => quote.classList.contains("quote-marked"))).toEqual([
    false,
    true,
  ]);
  // One line per document, its pages in order, each page a link.
  const pages = screen.getByRole("region", { name: "Pages to check yourself" });
  expect(within(pages).getAllByRole("listitem")).toHaveLength(2);
  const pageLinks = within(pages).getAllByRole("link");
  expect(pageLinks.map((link) => link.getAttribute("href"))).toEqual([
    `${report}#page=5`,
    `${report}#page=12`,
  ]);
  expect(pageLinks[1]).toHaveTextContent("p. 12 of Annual Report (opens in a new tab)");
  expect(within(pages).getByText("open")).toBeVisible();
  expect(within(pages).getByText(/A chart/)).toBeVisible();
  expect(
    screen.getByText(/Independent of NNPC Limited. Check the linked/),
  ).toBeVisible();
});

test("an advice request shows the fixed referral as information, with its steps listed", async () => {
  const { user, asked } = start();
  const referral = recorded(
    { understood_as: [] },
    {
      status: "safety_referral",
      notice:
        "It does not give investment, legal or safety advice.\n- For an investment decision, consult a licensed financial adviser.\n- In a safety emergency, call 112.",
      claims: [],
      sources: [],
    },
  );

  await askAndAnswer(user, asked, "Should I invest?", {
    kind: "answer",
    recorded: referral,
  });

  const panel = screen.getByRole("region", { name: "Advice referral" });
  expect(
    within(panel)
      .getAllByRole("listitem")
      .map((item) => item.textContent),
  ).toEqual([
    "For an investment decision, consult a licensed financial adviser.",
    "In a safety emergency, call 112.",
  ]);
  expect(screen.getByRole("status")).toHaveTextContent("Advice referral.");
});

test("a refusal lists the closest pages, and unknown hosts are not links", async () => {
  const { user, asked } = start();
  const refusal = recorded(
    { understood_as: ["Does NNPC deliver fuel to Kano?"] },
    {
      status: "insufficient_evidence",
      notice: "I could not find enough support.",
      claims: [],
      sources: [],
      closest_pages: [
        { title: "NETCO", url: "https://nnpcgroup.com/netco" },
        { title: "Elsewhere", url: "https://example.com/page" },
      ],
    },
  );

  await askAndAnswer(user, asked, "Delivery to Kano?", {
    kind: "answer",
    recorded: refusal,
  });

  const closest = screen.getByRole("region", { name: "Closest pages" });
  expect(within(closest).getByRole("link", { name: newTab("NETCO") })).toBeVisible();
  expect(within(closest).queryByRole("link", { name: /Elsewhere/ })).toBeNull();
  expect(within(closest).getByText("Elsewhere")).toBeVisible();
  expect(screen.getByText("Searched for")).toBeVisible();
  expect(screen.getByText("I could not find enough support.")).toBeVisible();
  expect(screen.queryByRole("button", { name: /Copy answer/ })).toBeNull();
  expect(screen.getAllByRole("status")[0]).toHaveTextContent(
    "Not enough information was found.",
  );
});

test("a refusal whose search only repeats the question does not repeat it", async () => {
  const { user, asked } = start();
  const refusal = recorded(
    { understood_as: ["Where does NETCO's school outreach take place?"] },
    {
      status: "insufficient_evidence",
      notice: "I could not find enough support.",
      claims: [],
      sources: [],
    },
  );

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: refusal });

  expect(screen.getByText("I could not find enough support.")).toBeVisible();
  expect(screen.queryByText("Searched for")).toBeNull();
});

test("a source off the allowed hosts shows its title and quote without links", async () => {
  const { user, asked } = start();
  const outside = recorded(
    {},
    {
      sources: [
        {
          number: 1,
          title: "Somewhere",
          heading: "Section",
          quote: "NETCO visits secondary schools.",
          link: "https://example.com/a",
          url: "https://example.com/a",
          published: "",
          captured: "2026-10-10",
          compiled: false,
        },
      ],
    },
  );

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: outside });

  const sources = screen.getByRole("region", { name: "Sources" });
  expect(within(sources).queryByRole("link")).toBeNull();
  expect(within(sources).getByText(/Somewhere/)).toBeVisible();
  expect(within(sources).getByText("› Section")).toBeVisible();
});

test("a quote from a list compiled from its page says so", async () => {
  const { user, asked } = start();
  const sources = view().sources.map((source) => ({ ...source, compiled: true }));
  const listed = recorded({}, { sources });

  await askAndAnswer(user, asked, "Which units?", { kind: "answer", recorded: listed });

  const shown = screen.getByRole("region", { name: "Sources" });
  expect(within(shown).getByText("Compiled from the page's list")).toBeVisible();
  expect(within(shown).queryByText("Found word for word")).toBeNull();
});

test("a failure shows the fixed message, and Try again asks the same question", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  await askAndAnswer(user, asked, "Who?", { kind: "error", detail: "Try later." });

  const alert = screen.getByRole("alert");
  expect(alert).toHaveTextContent("Try later.");
  expect(alert).toHaveClass("problem-error");
  await user.click(within(alert).getByRole("button", { name: "Try again" }));

  expect(asked[2]).toMatchObject({ question: "Who?", conversationId: "c-1" });
  // The retry takes the failed turn's place, and focus waits in the question box.
  expect(screen.getAllByRole("heading", { name: "Who?" })).toHaveLength(1);
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByLabelText("Your question")).toHaveFocus();
});

test("an answer copies with its sources, and a failed copy says so", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  await user.click(screen.getByRole("button", { name: "Copy answer with sources" }));

  expect(await screen.findByText("Copied with its sources.")).toBeVisible();
  const copied = await navigator.clipboard.readText();
  expect(copied).toContain("NETCO visits secondary schools. [1]");
  expect(copied).toContain("From PetroKnowledge, independent of NNPC Limited.");

  vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new Error("denied"));
  await user.click(screen.getByRole("button", { name: "Copy answer with sources" }));
  expect(
    await screen.findByText("Copying failed; select the text instead."),
  ).toBeVisible();
  vi.restoreAllMocks();
});

test("Try again on an earlier turn waits while another answer is on its way", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "error", detail: "Try later." });
  await user.type(screen.getByLabelText("Your question"), "Who?{Enter}");
  await user.click(screen.getByRole("button", { name: "Try again" }));

  expect(asked.map((one) => one.question)).toEqual(["Where?", "Who?"]);
});

test("Stop ends the answer on its way, which can then be asked again", async () => {
  const { user, asked } = start();

  await user.type(screen.getByLabelText("Your question"), "Where?{Enter}");
  expect(await screen.findByText(/Answering… Esc or Stop ends it/)).toBeVisible();
  const button = screen.getByRole("button", { name: "Stop" });
  await user.click(button);
  expect(asked[0]?.signal?.aborted).toBe(true);
  asked[0]?.send({ kind: "stopped" });
  asked[0]?.finish();

  expect(await screen.findByText("This answer did not finish.")).toBeVisible();
  // The same button now asks, and it kept keyboard focus (WCAG 2.4.3).
  expect(screen.getByRole("button", { name: "Ask" })).toBe(button);
  expect(button).toHaveFocus();
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent("Answer stopped.");
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(asked[1]?.question).toBe("Where?");
});

test("an ended conversation offers a new one, which clears the thread", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  await askAndAnswer(user, asked, "And NGIC?", {
    kind: "ended",
    detail: "It has ended.",
  });
  await user.click(screen.getByRole("button", { name: "Start a new conversation" }));

  expect(
    await screen.findByRole("heading", { name: "Ask a question to start" }),
  ).toBeVisible();
  expect(screen.getByLabelText("Your question")).toHaveFocus();
  await user.type(screen.getByLabelText("Your question"), "What is NGIC?{Enter}");
  expect(asked[2]?.conversationId).toBeNull();
});

test("New conversation forgets the conversation id, and stops an answer on its way", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  await user.type(screen.getByLabelText("Your question"), "Who?{Enter}");
  await user.click(screen.getByRole("button", { name: "New conversation" }));
  expect(asked[1]?.signal?.aborted).toBe(true);
  // Until the stopped answer has come back, nothing new is asked.
  await user.click(screen.getByRole("button", { name: "What does NNPC Shipping do?" }));
  expect(asked).toHaveLength(2);
  // The stopped answer arrives after the thread was cleared, and is dropped.
  asked[1]?.send({ kind: "stopped" });
  asked[1]?.finish();
  await screen.findByRole("heading", { name: "Ask a question to start" });

  await user.type(screen.getByLabelText("Your question"), "What is NGIC?{Enter}");
  expect(asked[2]?.conversationId).toBeNull();
});

test("Undo brings a cleared conversation back, until the next question", async () => {
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  await user.type(screen.getByLabelText("Your question"), "Who?{Enter}");
  await user.click(screen.getByRole("button", { name: "New conversation" }));
  expect(screen.getByRole("status")).toHaveTextContent(
    "Started a new conversation. Undo is available.",
  );
  await user.click(screen.getByRole("button", { name: "Undo" }));

  expect(screen.getByRole("heading", { name: "Where?" })).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("Conversation restored.");
  expect(screen.getByLabelText("Your question")).toHaveFocus();
  // The answer that was on its way when the thread was cleared did not finish.
  expect(screen.getByText("This answer did not finish.")).toBeVisible();
  asked[1]?.send({ kind: "stopped" });
  asked[1]?.finish();
  await screen.findByRole("button", { name: "Ask" });
  await user.type(screen.getByLabelText("Your question"), "And NGIC?{Enter}");
  expect(asked[2]?.conversationId).toBe("c-1");

  await user.click(screen.getByRole("button", { name: "New conversation" }));
  asked[2]?.send({ kind: "stopped" });
  asked[2]?.finish();
  await screen.findByRole("button", { name: "Ask" });
  await user.type(screen.getByLabelText("Your question"), "What is NGIC?{Enter}");
  expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
});

test("Esc stops the answer on its way wherever focus is, and does nothing otherwise", async () => {
  const { user, asked } = start();
  const box = screen.getByLabelText("Your question");

  await user.type(box, "draft{Escape}");
  expect(box).toHaveValue("draft");
  await user.clear(box);
  await user.type(box, "Where?{Enter}");
  await user.click(screen.getByRole("heading", { name: "Where?" }));
  await user.keyboard("{Escape}");

  expect(asked[0]?.signal?.aborted).toBe(true);
});

test("Stop with a draft in the box stops only, and the draft stays", async () => {
  const { user, asked } = start();
  const box = screen.getByLabelText("Your question");

  await user.type(box, "Where?{Enter}");
  await user.type(box, "and the follow-up");
  await user.click(screen.getByRole("button", { name: "Stop" }));
  asked[0]?.send({ kind: "stopped" });
  asked[0]?.finish();
  await screen.findByText("This answer did not finish.");

  expect(asked).toHaveLength(1);
  expect(box).toHaveValue("and the follow-up");
});

test("a reload keeps the thread, with an answer that was on its way marked stopped", async () => {
  sessionStorage.setItem(
    "petroknowledge.thread",
    JSON.stringify({
      conversationId: "c-9",
      turns: [{ key: 4, question: "Who leads NETCO?", stages: ["searching"] }],
    }),
  );
  const { user, asked } = start();

  expect(screen.getByRole("heading", { name: "Who leads NETCO?" })).toBeVisible();
  expect(screen.getByText("This answer did not finish.")).toBeVisible();
  await user.type(screen.getByLabelText("Your question"), "Since when?{Enter}");

  expect(asked[0]?.conversationId).toBe("c-9");
  expect(screen.getByRole("heading", { name: "Since when?" })).toBeVisible();
  const saved = JSON.parse(sessionStorage.getItem("petroknowledge.thread") ?? "{}");
  expect(saved.turns.map((turn: { key: number }) => turn.key)).toEqual([4, 5]);
});

test("an answer brings its question to the top unless the reader scrolled away", async () => {
  const scrolled = vi.spyOn(Element.prototype, "scrollIntoView");
  const { user, asked } = start();

  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });
  expect(scrolled).toHaveBeenLastCalledWith({ block: "start" });
  const callsAfterFirst = scrolled.mock.calls.length;

  await user.type(screen.getByLabelText("Your question"), "Who?{Enter}");
  const asking = scrolled.mock.calls.length;
  expect(asking).toBe(callsAfterFirst + 1);
  // The reader scrolled up: the new turn is now below the view.
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    top: window.innerHeight + 100,
  } as DOMRect);
  asked[1]?.send({ kind: "answer", recorded: recorded({ turn: 2 }) });
  asked[1]?.finish();
  await screen.findByRole("button", { name: "Ask" });

  expect(scrolled.mock.calls.length).toBe(asking);
  vi.restoreAllMocks();
});

test("the question box's height keeps scrolled-to content clear of it", () => {
  start();

  expect(document.documentElement.style.getPropertyValue("--composer-height")).toBe(
    "0px",
  );
});

test("without a stand-in, a question goes to the v1 stream", async () => {
  const fetchImpl = vi.fn(async () => new Response(null, { status: 503 }));
  vi.stubGlobal("fetch", fetchImpl);
  const user = userEvent.setup();
  render(ThreadApp);

  await user.type(screen.getByLabelText("Your question"), "Who leads NETCO?{Enter}");

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "The assistant can't answer right now.",
  );
  expect(fetchImpl).toHaveBeenCalledWith("/api/v1/answers/stream", expect.anything());
  vi.unstubAllGlobals();
});

test("Shift+Enter writes a new line, blank questions are not asked, and long ones show a count", async () => {
  const { user, asked } = start();
  const box = screen.getByLabelText("Your question");

  await user.type(box, "   {Enter}");
  await user.type(box, "First line{Shift>}{Enter}{/Shift}second line");
  expect(asked).toEqual([]);
  expect(box).toHaveValue("   First line\nsecond line");
  expect(screen.getByText(/Enter to ask/)).toBeVisible();

  await user.clear(box);
  await user.click(box);
  await user.paste("x".repeat(MAX_QUESTION_CHARS));
  expect(
    await screen.findByText(`${MAX_QUESTION_CHARS} / ${MAX_QUESTION_CHARS}`),
  ).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Ask" }));
  expect(asked[0]?.question).toHaveLength(MAX_QUESTION_CHARS);
});

test("a question is shown as text, never as markup", async () => {
  const { user, shown } = start();

  await user.type(
    screen.getByLabelText("Your question"),
    "<img src=x onerror=alert(1)>{Enter}",
  );

  expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeVisible();
  expect(shown.container.querySelector("img")).toBeNull();
});

test("a question waiting for the assistant shows its place in line, then its stages", async () => {
  const { user, asked } = start();

  await user.type(screen.getByLabelText("Your question"), "Where?{Enter}");
  asked[0]?.send({ kind: "queued", position: 2 });
  expect(
    await screen.findByText(
      /Other questions are being answered\. Yours is number 2 in line\./,
    ),
  ).toBeVisible();
  asked[0]?.send({ kind: "stage", stage: "understanding" });
  expect(await screen.findByText(/Reading the question/)).toBeVisible();
});

test("a reader can mark an answer right, or not right with a comment", async () => {
  const kept = vi.fn(async () => new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", kept);
  const { user, asked } = start();
  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });

  await user.click(screen.getByRole("button", { name: "Not right" }));
  const box = screen.getByLabelText("What was wrong? (optional)");
  expect(box).toHaveFocus();
  await user.type(box, "  wrong year  ");
  await user.click(screen.getByRole("button", { name: "Send" }));

  const thanks = await screen.findByText("Thanks. Your feedback was recorded.");
  expect(thanks).toHaveFocus();
  const [url, init] = kept.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("/api/v1/answers/7/feedback");
  expect(JSON.parse(init.body as string)).toEqual({
    conversation_id: "c-1",
    helpful: false,
    comment: "wrong year",
  });

  await askAndAnswer(user, asked, "And?", { kind: "answer", recorded: recorded() });
  await user.click(screen.getByRole("button", { name: "Yes" }));
  expect(
    await screen.findAllByText("Thanks. Your feedback was recorded."),
  ).toHaveLength(2);
  expect(
    JSON.parse(
      (kept.mock.calls[1] as unknown as [string, RequestInit])[1].body as string,
    ),
  ).toEqual({ conversation_id: "c-1", helpful: true, comment: "" });
  vi.unstubAllGlobals();
});

test("feedback that cannot be sent says so, and a second click while sending sends once", async () => {
  let finish: (response: Response) => void = () => {};
  const slow = vi.fn(() => new Promise<Response>((resolve) => (finish = resolve)));
  vi.stubGlobal("fetch", slow);
  const { user, asked } = start();
  await askAndAnswer(user, asked, "Where?", { kind: "answer", recorded: recorded() });

  await user.click(screen.getByRole("button", { name: "Yes" }));
  await user.click(screen.getByRole("button", { name: "Yes" }));
  finish(new Response(null, { status: 503 }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Feedback could not be sent. Try again.",
  );
  expect(slow).toHaveBeenCalledTimes(1);
  vi.unstubAllGlobals();
});
