import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

// The API is answered here: each test says what the stream returns, and the requests
// the page sent are kept for checking.

const SOURCE = {
  number: 1,
  title: "PRESS RELEASE: NNPC Delivers ₦7.2 Trillion Profit",
  heading: "",
  quote: "Profit After Tax rose 33% to ₦7.2 trillion from ₦5.4 trillion in 2024.",
  link: "https://nnpcgroup.com/posts/profit#:~:text=Profit%20After%20Tax",
  url: "https://nnpcgroup.com/posts/profit",
  published: "2026-09-29",
  captured: "2026-10-10",
  compiled: false,
};

function recorded(turn: number, change: Record<string, unknown> = {}) {
  return {
    id: 100 + turn,
    conversation_id: "6c1f0a52-0b7e-4c1d-9f3a-2d8b7e5a4c10",
    turn,
    understood_as: ["What profit after tax did NNPC report for 2025?"],
    answer: {
      question: "q",
      status: "answered",
      notice: "",
      claims: [
        {
          text: "NNPC Limited's profit after tax for 2025 was ₦7.2 trillion.",
          sources: [1],
        },
      ],
      sources: [SOURCE],
      closest_pages: [],
      pointers: [
        {
          title: "NNPC 2025 Annual Report",
          link: "https://cms1977.nnpcgroup.com/uploads/report.pdf#page=12",
          kind: "scanned page",
        },
      ],
      ...change,
    },
  };
}

function stream(answer: unknown): string {
  const stages = ["understanding", "searching", "answering", "checking"]
    .map((stage) => `event: stage\ndata: {"stage": "${stage}"}\n\n`)
    .join("");
  return `${stages}event: answer\ndata: ${JSON.stringify(answer)}\n\n`;
}

async function answerWith(page: Page, replies: ((route: Route) => Promise<void>)[]) {
  const sent: Record<string, unknown>[] = [];
  await page.route("**/api/v1/answers/stream", async (route) => {
    sent.push(route.request().postDataJSON() as Record<string, unknown>);
    const reply = replies[sent.length - 1] as (route: Route) => Promise<void>;
    await reply(route);
  });
  return sent;
}

function sse(answer: unknown) {
  return (route: Route) =>
    route.fulfill({ contentType: "text/event-stream", body: stream(answer) });
}

async function ask(page: Page, question: string) {
  await page.getByLabel("Your question").fill(question);
  await page.getByLabel("Your question").press("Enter");
}

async function expectAccessible(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(results.violations).toEqual([]);
}

/** Console errors, such as a script the security policy blocked. */
function consoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  return errors;
}

for (const colorScheme of ["light", "dark"] as const) {
  test(`a three-turn conversation, accessible in ${colorScheme}`, async ({
    page,
  }, info) => {
    const errors = consoleErrors(page);
    await page.emulateMedia({ colorScheme });
    const sent = await answerWith(page, [
      sse(recorded(1)),
      sse(
        recorded(2, {
          claims: [{ text: "Revenue stood at ₦34.5 trillion.", sources: [1] }],
        }),
      ),
      sse(
        recorded(3, {
          status: "safety_referral",
          notice:
            "This assistant gives information from NNPC Limited's public documents only. It does not give investment, legal or safety advice.\n- For an investment decision, consult a licensed financial adviser.\n- In a safety emergency, call 112.",
          claims: [],
          sources: [],
          pointers: [],
        }),
      ),
    ]);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask a question to start" }),
    ).toBeVisible();
    await expectAccessible(page);
    await page.screenshot({
      path: info.outputPath(`empty-${colorScheme}.png`),
      fullPage: true,
    });

    await ask(page, "What profit did NNPC make in 2025?");
    await expect(page.getByRole("link", { name: "Source 1" })).toBeVisible();
    await expect(page.getByText("Published 29 Sep 2026")).toBeVisible();
    await ask(page, "and its revenue?");
    await expect(page.getByText("Understood as")).toBeVisible();
    await ask(page, "should I buy NNPC bonds?");
    await expect(page.getByRole("region", { name: "Advice referral" })).toBeVisible();

    expect(sent.map((body) => body.conversation_id ?? null)).toEqual([
      null,
      "6c1f0a52-0b7e-4c1d-9f3a-2d8b7e5a4c10",
      "6c1f0a52-0b7e-4c1d-9f3a-2d8b7e5a4c10",
    ]);
    await expectAccessible(page);
    await page.screenshot({
      path: info.outputPath(`thread-${colorScheme}.png`),
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });
}

test("the page is served with its security headers and runs under them", async ({
  page,
}) => {
  const errors = consoleErrors(page);
  const response = await page.goto("/");

  const policy = response?.headers()["content-security-policy"] ?? "";
  expect(policy).toContain("default-src 'self'");
  expect(policy).toContain("frame-ancestors 'none'");
  expect(policy).not.toContain("unsafe-inline");
  await expect(page.getByRole("heading", { name: "PetroKnowledge" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("an ended conversation starts again, and a failure shows the fixed message", async ({
  page,
}) => {
  await answerWith(page, [
    (route) => route.fulfill({ status: 503, body: '{"detail": "x"}' }),
    sse(recorded(1)),
    (route) => route.fulfill({ status: 404, body: '{"detail": "ended"}' }),
  ]);
  await page.goto("/");

  await ask(page, "What profit did NNPC make in 2025?");
  await expect(page.getByRole("alert")).toContainText(
    "The assistant can't answer right now.",
  );
  await ask(page, "What profit did NNPC make in 2025?");
  await expect(page.getByRole("link", { name: "Source 1" })).toBeVisible();
  await ask(page, "And NGIC?");
  await page.getByRole("button", { name: "Start a new conversation" }).click();

  await expect(
    page.getByRole("heading", { name: "Ask a question to start" }),
  ).toBeVisible();
});

test("at phone width the thread fits without sideways scrolling", async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await answerWith(page, [sse(recorded(1))]);
  await page.goto("/");
  await ask(page, "What profit did NNPC make in 2025?");
  await expect(page.getByRole("link", { name: "Source 1" })).toBeVisible();
  // The answer starts in view: its question is at the top, not above the screen.
  const top = await page
    .getByRole("heading", { name: "What profit did NNPC make in 2025?" })
    .evaluate((heading) => heading.getBoundingClientRect().top);
  expect(top).toBeGreaterThanOrEqual(0);
  expect(top).toBeLessThan(200);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: info.outputPath("phone.png"), fullPage: true });
});

/** An answer long enough to scroll: several claims and sources. */
function longAnswer() {
  const sources = [1, 2, 3, 4, 5].map((number) => ({
    ...SOURCE,
    number,
    quote: `${SOURCE.quote} (${number})`,
    link: `${SOURCE.url}#:~:text=part${number}`,
  }));
  return recorded(1, {
    claims: [1, 2, 3].map((number) => ({
      text: `Statement ${number} about NNPC's results, verified against its source.`,
      sources: [number, number + 1],
    })),
    sources,
  });
}

test("keyboard focus is never hidden behind the question box (WCAG 2.4.11)", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await answerWith(page, [sse(longAnswer())]);
  await page.goto("/");
  await ask(page, "What profit did NNPC make in 2025?");
  await expect(page.getByRole("link", { name: "Source 1" }).first()).toBeVisible();

  await page.locator("h1").click();
  const hidden: string[] = [];
  for (let stop = 0; stop < 25; stop += 1) {
    await page.keyboard.press("Tab");
    const place = await page.evaluate(() => {
      const focused = document.activeElement as HTMLElement;
      const bar = document.querySelector(".composer-bar") as HTMLElement;
      if (bar.contains(focused)) return null;
      return {
        name: focused.textContent ?? "",
        top: focused.getBoundingClientRect().top,
        barTop: bar.getBoundingClientRect().top,
      };
    });
    if (place !== null && place.top >= place.barTop) hidden.push(place.name);
  }

  expect(hidden).toEqual([]);
});

test("at 400% zoom the question box scrolls with the page (WCAG 1.4.10)", async ({
  page,
}) => {
  // 1280x800 at 400% zoom is a 320x200 CSS-pixel viewport.
  await page.setViewportSize({ width: 320, height: 200 });
  await answerWith(page, [sse(recorded(1))]);
  await page.goto("/");
  await ask(page, "What profit did NNPC make in 2025?");
  await expect(page.getByRole("link", { name: "Source 1" })).toBeAttached();

  const position = await page
    .locator(".composer-bar")
    .evaluate((bar) => getComputedStyle(bar).position);
  expect(position).toBe("static");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
